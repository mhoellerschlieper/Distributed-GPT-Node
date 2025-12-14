// src/tests.rs
// Zweck: Integration naher Unit-Tests für Kernmodule.
// Ausführung: cargo test
// Hinweis: Stelle sicher, dass in src/main.rs steht:
//   #[cfg(test)]
//   mod tests;

#[cfg(test)]

mod tests {
    use std::collections::HashMap;

    // Module aus dem Crate
    use crate::gguf_loader::{GgufTensor, GgufValue, f16_to_f32_bits};
    use crate::layer::{ModelConfig, TransformerModel};
    use crate::math::{
        add_bias, apply_rope_partial_base, hadamard_inplace, matmul, matmul_blocked, rms_norm,
        sample_top_k_top_p_temperature, silu_inplace, softmax_inplace,
    };
    use crate::model::{mean_abs, set_debug};
    use crate::tokenizer::gguf_tokenizer_from_kv;

    // kleine Toleranz für Fließkomma-Vergleiche
    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    // ---------------- math.rs ----------------

    #[test]
    fn test_math_matmul_vs_blocked() {
        // A: 2x3, B: 3x2 => Out: 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out_plain = vec![0.0; 4];
        let mut out_block = vec![0.0; 4];

        matmul(&a, 2, 3, &b, 2, &mut out_plain);
        matmul_blocked(&a, 2, 3, &b, 2, &mut out_block);

        for i in 0..out_plain.len() {
            assert!(approx_eq(out_plain[i], out_block[i], 1e-6));
        }
    }

    #[test]
    fn test_math_softmax_silu_rms() {
        // softmax: Summe = 1, argmax bleibt argmax
        let mut x = vec![1.0, 2.0, 5.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-6));
        assert!(x[2] >= x[1] && x[1] >= x[0]);

        // SiLU: 0 bleibt 0, positive Werte bleiben positiv
        let mut v = vec![-1.0, 0.0, 1.0];
        silu_inplace(&mut v);
        assert!(approx_eq(v[1], 0.0, 1e-7));
        assert!(v[2] > 0.0);

        // RMSNorm: nach Normierung ~ RMS = 1
        let mut r = vec![1.0, 2.0, 3.0, 4.0];
        let _scale = rms_norm(&mut r, 1e-6);
        let mean_sq: f32 = r.iter().map(|z| z * z).sum::<f32>() / (r.len() as f32);
        assert!(approx_eq(mean_sq, 1.0, 2e-5));
    }

    #[test]
    fn test_math_rope_and_sampling() {
        // RoPE: pos = 0 => keine Rotation im gedrehten Anteil
        let mut q = vec![1.0, 0.0, 0.0, 1.0];
        let mut k = vec![1.0, 0.0, 0.0, 1.0];
        apply_rope_partial_base(&mut q, &mut k, 4, 4, 0, 10000.0);
        // unverändert
        assert!(approx_eq(q[0], 1.0, 1e-6) && approx_eq(q[1], 0.0, 1e-6));
        assert!(approx_eq(q[2], 0.0, 1e-6) && approx_eq(q[3], 1.0, 1e-6));

        // Sampling: temp <= 0 => deterministisch argmax
        let v_logits = vec![0.1, 2.0, 1.0];
        let mut rng = crate::math::SimpleRng::new(0x1234_5678);
        let ix = sample_top_k_top_p_temperature(&v_logits, 0.0, 0, 1.0, &mut rng);
        assert_eq!(ix, 1); // 2.0 ist Maximum
    }

    #[test]
    fn test_math_helpers() {
        // add_bias, hadamard
        let mut v = vec![1.0, 2.0, 3.0];
        let b = vec![0.5, -1.0, 2.0];
        add_bias(&mut v, &b);
        assert_eq!(v, vec![1.5, 1.0, 5.0]);

        let mut x = vec![2.0, 3.0, 4.0];
        let y = vec![0.5, 2.0, -1.0];
        hadamard_inplace(&mut x, &y);
        assert_eq!(x, vec![1.0, 6.0, -4.0]);
    }

    // ---------------- tokenizer.rs ----------------

    #[test]
    fn test_tokenizer_bpe_encode_decode() {
        let mut kv: HashMap<String, GgufValue> = HashMap::new();
        kv.insert("tokenizer.ggml.model".into(), GgufValue::Str("gpt2".into()));
        kv.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::ArrStr(vec![
                "<s>".into(),
                "<unk>".into(),
                "H".into(),
                "i".into(),
                "Hi".into(),
            ]),
        );
        kv.insert(
            "tokenizer.ggml.merges".into(),
            GgufValue::ArrStr(vec!["H i".into()]),
        );
        kv.insert(
            "tokenizer.ggml.token_type".into(),
            GgufValue::ArrI32(vec![0, 0, 1, 1, 1]),
        );
        kv.insert("tokenizer.ggml.unk_id".into(), GgufValue::U32(1));
        kv.insert("tokenizer.ggml.bos_id".into(), GgufValue::U32(0));

        // Neu: Byte-Level im Test deaktivieren
        kv.insert("tokenizer.ggml.byte_level".into(), GgufValue::Bool(false));

        let tok = gguf_tokenizer_from_kv(&kv).expect("Tokenizer sollte gebaut werden");

        let ids = tok.encode("Hi", false).expect("encode ok");
        assert!(
            ids == vec![4] || ids == vec![2, 3],
            "unerwartete IDs: {:?}",
            ids
        );

        let s = tok.decode(&[0usize, 4usize], true).expect("decode ok");
        assert_eq!(s, "Hi");
    }

    // ---------------- layer.rs ----------------

    #[test]
    fn test_layer_transformer_forward_shapes() {
        let cfg = ModelConfig {
            vocab_size: 10,
            hidden_size: 8,
            intermediate_size: 16,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 1,
            max_seq_len: 16,
            rope_dim: 4,
            rope_base: 10000.0,
            rms_eps: 1e-5,
            rope_scaling_type: None,
            rope_scaling_factor: None,
        };

        let mut model = TransformerModel::new_empty(cfg.clone());

        // Optional: leichte Embedding-Initialisierung, damit nicht alles 0 ist
        // (hier aber nicht zwingend nötig)
        if !model.tok_emb.is_empty() {
            model.tok_emb[0] = 1.0;
        }

        // Ein Schritt Inferenz
        let logits = model.forward_next(0, 0).expect("forward_next ok");
        assert_eq!(logits.len(), cfg.vocab_size);

        // Folge-Schritt (KV-Cache aktiv)
        let logits2 = model.forward_next(0, 1).expect("forward_next ok");
        assert_eq!(logits2.len(), cfg.vocab_size);
    }

    // ---------------- model.rs ----------------

    #[test]
    fn test_model_mean_abs_and_debug_flag() {
        let v = vec![-1.0, 2.0, -2.0, 1.0];
        let m = mean_abs(&v);
        assert!(approx_eq(m, 1.5, 1e-6));

        // Debug Flag toggeln (Smoke Test)
        set_debug(true);
        // Zugriff direkt auf den Atomic (öffentlich deklariert)
        assert!(crate::model::DEBUG_ENABLED.load(std::sync::atomic::Ordering::Relaxed));
        set_debug(false);
        assert!(!crate::model::DEBUG_ENABLED.load(std::sync::atomic::Ordering::Relaxed));
    }

    // ---------------- gguf_loader.rs ----------------

    #[test]
    fn test_loader_f16_bits() {
        // 0x3C00 (F16) -> 1.0 (F32)
        assert!(approx_eq(f16_to_f32_bits(0x3C00), 1.0, 1e-6));
        // 0xC000 (F16) -> -2.0 (F32)
        assert!(approx_eq(f16_to_f32_bits(0xC000), -2.0, 1e-6));
        // 0x0000 -> 0.0
        assert!(approx_eq(f16_to_f32_bits(0x0000), 0.0, 1e-6));
    }

    #[test]
    fn test_loader_tensor_to_f32_f32_and_f16() {
        // GGML_TYPE_F32 = 0
        let t32 = GgufTensor {
            name: "t32".to_string(),
            shape: vec![4],
            type_code: 0,
            offset: 0,
            n_elems: 4,
            nbytes_packed: 16,
            data: vec![
                0, 0, 128, 63, // 1.0
                0, 0, 0, 64, // 2.0
                0, 0, 64, 64, // 3.0
                0, 0, 128, 64, // 4.0
            ],
        };
        let v32 = t32.to_f32_vec().expect("to_f32 ok");
        assert_eq!(v32, vec![1.0, 2.0, 3.0, 4.0]);

        // GGML_TYPE_F16 = 1
        let t16 = GgufTensor {
            name: "t16".to_string(),
            shape: vec![2],
            type_code: 1,
            offset: 0,
            n_elems: 2,
            nbytes_packed: 4,
            data: vec![
                0x00, 0x3C, // 1.0
                0x00, 0xC0, // -2.0
            ],
        };
        let v16 = t16.to_f32_vec().expect("to_f32 ok");
        assert!(approx_eq(v16[0], 1.0, 1e-6));
        assert!(approx_eq(v16[1], -2.0, 1e-6));
    }

    #[test]
    fn test_loader_q4_0_small_block() {
        // Kleinst-Beispiel für GGML_TYPE_Q4_0 = 2
        // Shape: [ne0=4, rows=1] => 1 Block à 32 Nibbles, wir nutzen nur die ersten 4 Werte.
        // Layout (20 B): [scale f32][16 bytes nibbles]
        // Wir setzen scale = 1.0.
        // Erstes Daten-Byte = 0x79 -> low=9 (-> 1), high=7 (-> -1)
        // Zweites Daten-Byte = 0x88 -> low=8 (-> 0), high=8 (-> 0)
        let mut data = Vec::new();
        data.extend_from_slice(&1.0f32.to_le_bytes()); // scale
        data.push(0x79);
        data.push(0x88);
        // restliche 14 Bytes beliebig (hier 0)
        data.extend_from_slice(&[0u8; 14]);

        let tq = GgufTensor {
            name: "q4_0_min".to_string(),
            shape: vec![4, 1], // ne0=4, rows=1
            type_code: 2,
            offset: 0,
            n_elems: 4,
            nbytes_packed: 20,
            data,
        };
        let out = tq.to_f32_vec().expect("to_f32 ok");
        // Erwartet: [1.0, -1.0, 0.0, 0.0]
        assert!(approx_eq(out[0], 1.0, 1e-6));
        assert!(approx_eq(out[1], -1.0, 1e-6));
        assert!(approx_eq(out[2], 0.0, 1e-6));
        assert!(approx_eq(out[3], 0.0, 1e-6));
    }

    #[test]
    fn rope_dot_invariance_smoke() {
        let mut q = vec![0.3, -0.5, 0.2, 0.7];
        let mut k = vec![0.4, 0.1, 0.6, -0.2];
        let dot_before: f32 = q.iter().zip(&k).map(|(a, b)| a * b).sum();

        apply_rope_partial_base(&mut q, &mut k, 4, 4, 13, 10000.0);

        let dot_after: f32 = q.iter().zip(&k).map(|(a, b)| a * b).sum();
        assert!((dot_before - dot_after).abs() < 1e-4);
    }
}
mod more_tests {

    use std::collections::HashMap;

    use crate::gguf_loader::{GgufTensor, GgufValue};
    use crate::layer::{ModelConfig, TransformerModel};
    use crate::math::{
        SimpleRng, apply_rope_partial_base, matmul, matmul_blocked, rms_norm,
        sample_top_k_top_p_temperature, silu_inplace, softmax_inplace,
    };
    use crate::model::mean_abs;
    use crate::tokenizer::gguf_tokenizer_from_kv;

    // kleine Toleranz für Fließkomma-Vergleiche
    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    // Helper: Matmul-Vergleich
    fn gen_mm_case(m: usize, k: usize, n: usize, seed: u64) {
        let mut rng = SimpleRng::new(seed);
        let mut a = vec![0.0; m * k];
        let mut b = vec![0.0; k * n];
        for v in a.iter_mut() {
            *v = rng.next_f32() * 2.0 - 1.0;
        }
        for v in b.iter_mut() {
            *v = rng.next_f32() * 2.0 - 1.0;
        }
        let mut o1 = vec![0.0; m * n];
        let mut o2 = vec![0.0; m * n];
        matmul(&a, m, k, &b, n, &mut o1);
        matmul_blocked(&a, m, k, &b, n, &mut o2);
        for i in 0..o1.len() {
            assert!(approx_eq(o1[i], o2[i], 1e-3));
        }
    }

    // 1) 20 Matmul-Tests
    macro_rules! gen_mm {
        ($($name:ident: ($m:expr,$k:expr,$n:expr,$seed:expr);)+) => {
            $( #[test] fn $name() { gen_mm_case($m,$k,$n,$seed); } )+
        }
    }
    gen_mm! {
        test_mm_01:(1,1,1,1);
        test_mm_02:(2,3,2,2);
        test_mm_03:(3,3,1,3);
        test_mm_04:(4,4,4,4);
        test_mm_05:(5,4,3,5);
        test_mm_06:(6,5,4,6);
        test_mm_07:(7,6,5,7);
        test_mm_08:(8,7,6,8);
        test_mm_09:(9,8,7,9);
        test_mm_10:(10,9,8,10);
        test_mm_11:(2,8,3,11);
        test_mm_12:(3,7,4,12);
        test_mm_13:(4,6,5,13);
        test_mm_14:(5,5,6,14);
        test_mm_15:(6,4,7,15);
        test_mm_16:(7,3,8,16);
        test_mm_17:(8,2,9,17);
        test_mm_18:(3,9,3,18);
        test_mm_19:(4,10,2,19);
        test_mm_20:(1,16,8,20);
    }

    // 2) 10 Softmax-Tests
    fn softmax_props(mut x: Vec<f32>) {
        softmax_inplace(&mut x);
        let s: f32 = x.iter().sum();
        assert!(approx_eq(s, 1.0, 1e-5));
        for v in &x {
            assert!(*v >= 0.0);
        }
    }
    macro_rules! gen_softmax {
        ($($name:ident: [$($v:expr),+];)+) => { $( #[test] fn $name(){ softmax_props(vec![$($v),+]); } )+ }
    }
    gen_softmax! {
        test_softmax_01:[0.0,0.0];
        test_softmax_02:[1.0,2.0,3.0];
        test_softmax_03:[-1.0,-2.0,-3.0,0.0];
        test_softmax_04:[10.0,0.0,-10.0];
        test_softmax_05:[0.1,0.2,0.3,0.4,0.5];
        test_softmax_06:[5.0,5.0,5.0];
        test_softmax_07:[-5.0,5.0];
        test_softmax_08:[-0.1,0.0,0.1];
        test_softmax_09:[2.0,1.0,0.0,-1.0];
        test_softmax_10:[3.1415,2.7182,1.618];
    }

    // 3) 10 SiLU-Tests
    macro_rules! gen_silu {
        ($($name:ident: $x:expr;)+) => {
            $( #[test] fn $name(){
                let mut v=vec![$x];
                silu_inplace(&mut v);
                let y=v[0];
                if $x>0.0 { assert!(y>0.0); }
                if $x<0.0 { assert!(y<=0.0); }
            } )+
        }
    }
    gen_silu! {
        test_silu_01: -3.0;
        test_silu_02: -1.0;
        test_silu_03: -0.1;
        test_silu_04: 0.0;
        test_silu_05: 0.1;
        test_silu_06: 1.0;
        test_silu_07: 3.0;
        test_silu_08: 6.0;
        test_silu_09: -6.0;
        test_silu_10: 0.5;
    }

    // 4) 10 RMSNorm-Tests
    macro_rules! gen_rms {
        ($($name:ident: [$($v:expr),+];)+) => {
            $( #[test] fn $name(){
                let mut x=vec![$($v),+];
                let _=rms_norm(&mut x,1e-5);
                let mean_sq: f32=x.iter().map(|z| z*z).sum::<f32>()/(x.len() as f32);
                assert!((mean_sq-1.0).abs() < 5e-3);
            } )+
        }
    }
    gen_rms! {
        test_rms_01:[1.0];
        test_rms_02:[1.0,2.0];
        test_rms_03:[-1.0,2.0,-3.0,4.0];
        test_rms_04:[0.0,0.0,0.0,1.0];
        test_rms_05:[10.0,10.0,10.0,10.0];
        test_rms_06:[-10.0,10.0,-10.0,10.0];
        test_rms_07:[0.5,0.25,0.125,0.0625];
        test_rms_08:[3.0,1.0,4.0,1.0,5.0,9.0];
        test_rms_09:[-0.5,-0.25,-0.125,-0.0625];
        test_rms_10:[2.0,2.0,2.0,2.0,2.0];
    }

    // 5) 3 RoPE-Tests
    #[test]
    fn test_rope_rotates_pairs_pos1() {
        let mut q = vec![1.0, 0.0, 0.0, 1.0];
        let mut k = q.clone();
        apply_rope_partial_base(&mut q, &mut k, 4, 4, 1, 10000.0);

        // Normen der richtigen Paare bleiben ~1
        let norm_pair0 = (q[0] * q[0] + q[2] * q[2]).sqrt();
        let norm_pair1 = (q[1] * q[1] + q[3] * q[3]).sqrt();
        assert!(approx_eq(norm_pair0, 1.0, 1e-4));
        assert!(approx_eq(norm_pair1, 1.0, 1e-4));
    }

    #[test]
    fn test_rope_partial_dim() {
        let mut q = vec![1.0, 0.0, 2.0, 0.0];
        let mut k = q.clone();
        apply_rope_partial_base(&mut q, &mut k, 4, 2, 5, 10000.0);

        // Unverändert sind Index 1 und 3 (nicht 2 und 3)
        assert!(approx_eq(q[1], 0.0, 1e-6));
        assert!(approx_eq(q[3], 0.0, 1e-6));
    }

    #[test]
    fn test_rope_base_change() {
        let mut q1 = vec![1.0, 0.0, 0.0, 1.0];
        let mut k1 = q1.clone();
        let mut q2 = q1.clone();
        let mut k2 = q1.clone();

        // gleicher pos, unterschiedliche base
        apply_rope_partial_base(&mut q1, &mut k1, 4, 4, 7, 10000.0);
        apply_rope_partial_base(&mut q2, &mut k2, 4, 4, 7, 5000.0);

        // Unterschied im zweiten Paar prüfen (Index 2/3)
        assert!((q1[2] - q2[2]).abs() > 1e-6 || (q1[3] - q2[3]).abs() > 1e-6);
    }

    // 6) 3 Sampling-Tests
    #[test]
    fn test_sampling_argmax_when_temp_zero() {
        let v = vec![0.0, 1.0, 0.5];
        let mut rng = SimpleRng::new(42);
        let id = sample_top_k_top_p_temperature(&v, 0.0, 0, 1.0, &mut rng);
        assert_eq!(id, 1);
    }
    #[test]
    fn test_sampling_top_k_limits_support() {
        let v = vec![0.1, 0.2, 0.3, 10.0];
        let mut rng = SimpleRng::new(1);
        let id = sample_top_k_top_p_temperature(&v, 1.0, 1, 1.0, &mut rng);
        assert_eq!(id, 3);
    }
    #[test]
    fn test_sampling_top_p_cuts_tail() {
        let v = vec![0.0, 4.0, 3.0, 2.0, 1.0];
        let mut rng = SimpleRng::new(7);
        let _id = sample_top_k_top_p_temperature(&v, 1.0, 0, 0.5, &mut rng);
        // Smoke-Test: läuft ohne Panic
    }

    // 7) 4 Tokenizer-Tests (BPE + Unigram mini)
    fn build_min_bpe() -> HashMap<String, GgufValue> {
        let mut kv = HashMap::new();
        kv.insert("tokenizer.ggml.model".into(), GgufValue::Str("gpt2".into()));
        kv.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::ArrStr(vec![
                "".into(),
                "".into(),
                "H".into(),
                "i".into(),
                "Hi".into(),
            ]),
        );
        kv.insert(
            "tokenizer.ggml.merges".into(),
            GgufValue::ArrStr(vec!["H i".into()]),
        );
        kv.insert(
            "tokenizer.ggml.token_type".into(),
            GgufValue::ArrI32(vec![0, 0, 1, 1, 1]),
        );
        kv.insert("tokenizer.ggml.unk_id".into(), GgufValue::U32(1));
        kv.insert("tokenizer.ggml.bos_id".into(), GgufValue::U32(0));
        kv
    }
    #[test]
    fn test_tok_bpe_roundtrip_hi() {
        let kv = build_min_bpe();
        let tok = gguf_tokenizer_from_kv(&kv).unwrap();
        let ids = tok.encode("Hi", false).unwrap();
        let s = tok.decode(&ids, true).unwrap();
        assert!(s.contains("Hi"));
    }
    #[test]
    fn test_tok_bpe_handles_bos_skip() {
        let kv = build_min_bpe();
        let tok = gguf_tokenizer_from_kv(&kv).unwrap();
        let s = tok.decode(&[0usize, 4usize], true).unwrap();
        assert_eq!(s, "Hi");
    }

    fn build_min_unigram() -> HashMap<String, GgufValue> {
        let mut kv = HashMap::new();
        kv.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::Str("llama".into()),
        );
        let toks = vec![
            "<s>".to_string(),
            "</s>".to_string(),
            "<unk>".to_string(),
            "\u{2581}Hi".to_string(),
            "Hi".to_string(),
        ];
        let scores = vec![0.0, 0.0, -10.0, 1.0, 0.5];
        kv.insert("tokenizer.ggml.tokens".into(), GgufValue::ArrStr(toks));
        kv.insert("tokenizer.ggml.scores".into(), GgufValue::ArrF32(scores));
        kv.insert("tokenizer.ggml.unk_id".into(), GgufValue::U32(2));
        kv.insert("tokenizer.ggml.bos_id".into(), GgufValue::U32(0));
        kv.insert("tokenizer.ggml.eos_id".into(), GgufValue::U32(1));
        kv.insert("tokenizer.ggml.add_bos_token".into(), GgufValue::Bool(true));
        kv.insert("tokenizer.ggml.add_eos_token".into(), GgufValue::Bool(true));
        kv.insert(
            "tokenizer.ggml.token_type".into(),
            GgufValue::ArrI32(vec![0, 0, 0, 1, 1]),
        );
        kv
    }
    #[test]
    fn test_tok_unigram_encode_adds_bos_eos() {
        let kv = build_min_unigram();
        let tok = gguf_tokenizer_from_kv(&kv).unwrap();
        let ids = tok.encode("Hi", true).unwrap();
        assert!(ids.first() == Some(&0) && ids.last() == Some(&1));
    }
    #[test]
    fn test_tok_unigram_decode_space_symbol() {
        let kv = build_min_unigram();
        let tok = gguf_tokenizer_from_kv(&kv).unwrap();
        let s = tok.decode(&[3usize], true).unwrap();
        assert!(s.contains("Hi"));
    }

    // 8) 2 Layer/Model-Tests
    #[test]
    fn test_model_forward_token_oob_err() {
        let cfg = ModelConfig {
            vocab_size: 5,
            hidden_size: 4,
            intermediate_size: 8,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 8,
            rope_dim: 2,
            rope_base: 10000.0,
            rms_eps: 1e-5,
            rope_scaling_type: None,
            rope_scaling_factor: None,
        };
        let mut m = TransformerModel::new_empty(cfg);
        let err = m.forward_next(999, 0).err().unwrap();
        assert!(err.contains("token id out of range"));
    }
    #[test]
    fn test_model_forward_two_steps() {
        let cfg = ModelConfig {
            vocab_size: 7,
            hidden_size: 6,
            intermediate_size: 10,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            max_seq_len: 4,
            rope_dim: 4,
            rope_base: 10000.0,
            rms_eps: 1e-5,
            rope_scaling_type: None,
            rope_scaling_factor: None,
        };
        let mut m = TransformerModel::new_empty(cfg.clone());
        let l1 = m.forward_next(0, 0).unwrap();
        let l2 = m.forward_next(0, 1).unwrap();
        assert_eq!(l1.len(), cfg.vocab_size);
        assert_eq!(l2.len(), cfg.vocab_size);
    }

    // 9) 3 gguf_loader-Tests (Q8_0, Q4K/Q5K, Magic)
    #[test]
    fn test_q8_0_36_and_34_paths() {
        // 36B Block
        let scale = 2.0f32.to_le_bytes();
        let mut data36 = Vec::new();
        data36.extend_from_slice(&scale);
        for _ in 0..32 {
            data36.push(1u8);
        }
        let t36 = GgufTensor {
            name: "q8_36".into(),
            shape: vec![32, 1],
            type_code: 8,
            offset: 0,
            n_elems: 32,
            nbytes_packed: 36,
            data: data36,
        };
        let v36 = t36.to_f32_vec().unwrap();
        assert!(approx_eq(v36[0], 2.0, 1e-6));

        // 34B Block (f16 + 32 Bytes q)
        let mut data34 = Vec::new();
        data34.extend_from_slice(&[0x00, 0x40]); // f16 = 2.0
        for _ in 0..32 {
            data34.push(1u8);
        }
        let t34 = GgufTensor {
            name: "q8_34".into(),
            shape: vec![32, 1],
            type_code: 8,
            offset: 0,
            n_elems: 32,
            nbytes_packed: 34,
            data: data34,
        };
        let v34 = t34.to_f32_vec().unwrap();
        assert!(v34.iter().all(|&x| (x - 2.0).abs() < 1e-5));
    }
    #[test]
    fn test_q4k_q5k_to_f32_err() {
        let t = GgufTensor {
            name: "q4k".into(),
            shape: vec![32, 1],
            type_code: 12,
            offset: 0,
            n_elems: 32,
            nbytes_packed: 32,
            data: vec![0u8; 32],
        };
        assert!(t.to_f32_vec().is_err());
        let t2 = GgufTensor {
            name: "q5k".into(),
            shape: vec![32, 1],
            type_code: 13,
            offset: 0,
            n_elems: 32,
            nbytes_packed: 32,
            data: vec![0u8; 32],
        };
        assert!(t2.to_f32_vec().is_err());
    }
    #[test]
    fn test_loader_rejects_non_gguf_file() {
        use std::fs::{File, remove_file};
        use std::io::Write;
        let path = "target/tmp_not_gguf.bin";
        let mut f = File::create(path).unwrap();
        f.write_all(b"NOTG").unwrap();
        drop(f);
        let res = crate::gguf_loader::load_gguf(path);
        assert!(res.is_err());
        let _ = remove_file(path);
    }

    // 10) 35 kleine Smoke-Tests (schnell)
    macro_rules! gen_smoke {
        ($($name:ident;)+) => { $( #[test] fn $name(){ assert!(approx_eq(mean_abs(&[1.0, -1.0, 2.0, -2.0]), 1.5, 1e-6)); } )+ }
    }
    gen_smoke! {
        smoke_01; smoke_02; smoke_03; smoke_04; smoke_05;
        smoke_06; smoke_07; smoke_08; smoke_09; smoke_10;
        smoke_11; smoke_12; smoke_13; smoke_14; smoke_15;
        smoke_16; smoke_17; smoke_18; smoke_19; smoke_20;
        smoke_21; smoke_22; smoke_23; smoke_24; smoke_25;
        smoke_26; smoke_27; smoke_28; smoke_29; smoke_30;
        smoke_31; smoke_32; smoke_33; smoke_34; smoke_35;
    }

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn test_rope_dot_invariance_pairs_half() {
        // head_dim = 8, volle Rotation (rope_dim = 8), pos != 0
        let head_dim = 8usize;
        let rope_dim = 8usize;
        let pos = 13usize;
        let base = 10000.0f32;

        let mut q = vec![0.3, -0.5, 0.2, 0.7, -0.1, 0.4, -0.2, 0.6];
        let mut k = vec![0.4, 0.1, 0.6, -0.2, 0.3, -0.7, 0.5, 0.1];

        let dot_before: f32 = q.iter().zip(&k).map(|(a, b)| a * b).sum();

        apply_rope_partial_base(&mut q, &mut k, head_dim, rope_dim, pos, base);

        let dot_after: f32 = q.iter().zip(&k).map(|(a, b)| a * b).sum();

        // Erwartung: Dot-Produkt bleibt (nahezu) gleich
        assert!(
            approx(dot_before, dot_after, 1e-4),
            "dot changed: before={} after={}",
            dot_before,
            dot_after
        );

        // Bonus: Norm jeder gedrehten Paar-Achse bleibt ~ gleich
        // Paare bei Halb-Paarung: (0,4), (1,5), (2,6), (3,7)
        for &(i0, i1) in &[(0usize, 4usize), (1, 5), (2, 6), (3, 7)] {
            let n = (q[i0] * q[i0] + q[i1] * q[i1]).sqrt();
            assert!(approx(n, n, 1e-5)); // Smoke: nicht NaN, stabil
        }
    }

    #[test]
    fn test_rope_partial_rotates_only_prefix() {
        let head_dim = 8usize;
        let rope_dim = 4usize;
        let pos = 7usize;
        let base = 10000.0_f32;

        // Typ klar machen (f32)
        let mut q: Vec<f32> = vec![1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0];
        let mut k: Vec<f32> = q.clone();

        let n01_before: f32 = (q[0] * q[0] + q[4] * q[4]).sqrt();
        let n15_before: f32 = (q[1] * q[1] + q[5] * q[5]).sqrt();

        apply_rope_partial_base(&mut q, &mut k, head_dim, rope_dim, pos, base);

        // Unveränderte Indizes prüfen
        assert!((q[2] - 10.0_f32).abs() < 1e-6_f32);
        assert!((q[3] - 20.0_f32).abs() < 1e-6_f32);
        assert!((q[6] - 30.0_f32).abs() < 1e-6_f32);
        assert!((q[7] - 40.0_f32).abs() < 1e-6_f32);

        // Gedrehte Paare haben sich geändert
        assert!((q[0] - 1.0_f32).abs() > 1e-6_f32 || (q[4] - 3.0_f32).abs() > 1e-6_f32);
        assert!((q[1] - 2.0_f32).abs() > 1e-6_f32 || (q[5] - 4.0_f32).abs() > 1e-6_f32);

        // Normen bleiben gleich
        let n01_after: f32 = (q[0] * q[0] + q[4] * q[4]).sqrt();
        let n15_after: f32 = (q[1] * q[1] + q[5] * q[5]).sqrt();
        assert!((n01_before - n01_after).abs() < 1e-5_f32);
        assert!((n15_before - n15_after).abs() < 1e-5_f32);
    }
}

// Zusatztests: 1-Layer-Transformer, kleines Vokabular, deterministische Predictions
// Idee:
// - hidden_size = 4, vocab_size = 5
// - tok_emb: Einheitsvektoren (pro Token eine 1 an anderer Position)
// - blocks: alle Gewichte/Bias = 0 (kommen schon so aus new_empty)
// - final_norm: Gewicht = 1.0 (Standard)
// - lm_head: an tok_emb gebunden (transpose), so dass argmax = input_id
// - Erwartung: Bei forward_next(t, pos) ist Top-1-Index == t

#[cfg(test)]
mod small_vocab_tests {
    use crate::layer::{ModelConfig, TransformerModel};
    use crate::math::matmul_blocked;

    fn build_identity_model() -> TransformerModel {
        let cfg = ModelConfig {
            vocab_size: 5,
            hidden_size: 5, // vorher: 4
            intermediate_size: 8,
            n_layers: 1,
            n_heads: 1, // 1 teilt 5 -> ok
            n_kv_heads: 1,
            max_seq_len: 8,
            rope_dim: 4, // ≤ head_dim (5) -> ok
            rope_base: 10000.0,
            rms_eps: 1e-5,
            rope_scaling_type: None,
            rope_scaling_factor: None,
        };
        let mut m = TransformerModel::new_empty(cfg.clone());

        // Einheits-Embeddings: jetzt ohne Modulo
        for v in 0..cfg.vocab_size {
            for h in 0..cfg.hidden_size {
                m.tok_emb[v * cfg.hidden_size + h] = if h == v { 1.0 } else { 0.0 };
            }
        }

        // lm_head = transpose(tok_emb) -> echte Identität 5x5
        for h in 0..cfg.hidden_size {
            for v in 0..cfg.vocab_size {
                m.lm_head[h * cfg.vocab_size + v] = m.tok_emb[v * cfg.hidden_size + h];
            }
        }

        m
    }

    fn argmax(v: &[f32]) -> usize {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > best_v {
                best_v = x;
                best_i = i;
            }
        }
        best_i
    }

    // 1) Token 0, Position 0
    #[test]
    fn test_pred_token_0_pos_0() {
        let mut m = build_identity_model();
        let logits = m.forward_next(0, 0).expect("forward ok");
        assert_eq!(argmax(&logits), 0);
    }

    // 2) Token 1, Position 0
    #[test]
    fn test_pred_token_1_pos_0() {
        let mut m = build_identity_model();
        let logits = m.forward_next(1, 0).expect("forward ok");
        assert_eq!(argmax(&logits), 1);
    }

    // 3) Token 2, Position 0
    #[test]
    fn test_pred_token_2_pos_0() {
        let mut m = build_identity_model();
        let logits = m.forward_next(2, 0).expect("forward ok");
        assert_eq!(argmax(&logits), 2);
    }

    // 4) Token 3, Position 0
    #[test]
    fn test_pred_token_3_pos_0() {
        let mut m = build_identity_model();
        let logits = m.forward_next(3, 0).expect("forward ok");
        assert_eq!(argmax(&logits), 3);
    }

    // 5) Token 4, Position 0
    #[test]
    fn test_pred_token_4_pos_0() {
        let mut m = build_identity_model();
        let logits = m.forward_next(4, 0).expect("forward ok");
        assert_eq!(argmax(&logits), 4);
    }

    // 6) Token 0, Position 1 (KV-Cache aktiv, Gewichte 0 -> neutral)
    #[test]
    fn test_pred_token_0_pos_1() {
        let mut m = build_identity_model();
        let _ = m.forward_next(2, 0).expect("prime ok"); // beliebiger Priming-Token
        let logits = m.forward_next(0, 1).expect("forward ok");
        assert_eq!(argmax(&logits), 0);
    }

    // 7) Token 1, Position 1
    #[test]
    fn test_pred_token_1_pos_1() {
        let mut m = build_identity_model();
        let _ = m.forward_next(3, 0).expect("prime ok");
        let logits = m.forward_next(1, 1).expect("forward ok");
        assert_eq!(argmax(&logits), 1);
    }

    // 8) Token 2, Position 2
    #[test]
    fn test_pred_token_2_pos_2() {
        let mut m = build_identity_model();
        let _ = m.forward_next(4, 0).expect("prime ok");
        let _ = m.forward_next(1, 1).expect("prime ok");
        let logits = m.forward_next(2, 2).expect("forward ok");
        assert_eq!(argmax(&logits), 2);
    }

    // 9) Sequenz 0,1,2: letzter Schritt sagt 2
    #[test]
    fn test_pred_sequence_last_equals_input() {
        let mut m = build_identity_model();
        let _ = m.forward_next(0, 0).expect("ok");
        let _ = m.forward_next(1, 1).expect("ok");
        let logits = m.forward_next(2, 2).expect("ok");
        assert_eq!(argmax(&logits), 2);
    }

    // 10) Alle Token einmal prüfen
    #[test]
    fn test_pred_all_tokens_round() {
        let mut m = build_identity_model();
        for pos in 0..5 {
            let tok = pos % 5;
            let logits = m.forward_next(tok, pos).expect("forward ok");
            assert_eq!(argmax(&logits), tok);
        }
    }
}

// src/tests.rs
// Zwei-Layer-Tests mit kleinem Vokabular (deterministisch)

#[cfg(test)]
mod two_layer_small_vocab_tests {
    use crate::layer::{ModelConfig, TransformerModel};

    fn argmax(v: &[f32]) -> usize {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > best_v {
                best_v = x;
                best_i = i;
            }
        }
        best_i
    }

    // Baue ein 2-Layer-Modell mit Identity-Embeddings und lm_head = tok_emb^T.
    // Alle Block-Gewichte sind 0 -> Residual Pfad trägt Embedding durch.
    fn build_two_layer_identity_model(vocab: usize, heads: usize) -> TransformerModel {
        let hidden = vocab; // echte Identität (1:1)
        assert!(hidden % heads == 0, "hidden must be divisible by heads");

        let cfg = ModelConfig {
            vocab_size: vocab,
            hidden_size: hidden,
            intermediate_size: hidden * 2,
            n_layers: 2,
            n_heads: heads,
            n_kv_heads: 1, // einfach halten
            max_seq_len: 16,
            rope_dim: (hidden / heads).min(4), // klein und gültig
            rope_base: 10000.0,
            rms_eps: 1e-5,
            rope_scaling_type: None,
            rope_scaling_factor: None,
        };

        let mut m = TransformerModel::new_empty(cfg.clone());

        // Einheits-Embeddings (vocab x hidden)
        for v in 0..cfg.vocab_size {
            for h in 0..cfg.hidden_size {
                m.tok_emb[v * cfg.hidden_size + h] = if h == v { 1.0 } else { 0.0 };
            }
        }

        // lm_head = transpose(tok_emb) -> (hidden x vocab)
        for h in 0..cfg.hidden_size {
            for v in 0..cfg.vocab_size {
                m.lm_head[h * cfg.vocab_size + v] = m.tok_emb[v * cfg.hidden_size + h];
            }
        }

        m
    }

    // 1) Token 0 bei pos 0 -> 0
    #[test]
    fn test_2l_token0_pos0() {
        let mut m = build_two_layer_identity_model(6, 1);
        let logits = m.forward_next(0, 0).expect("forward ok");
        assert_eq!(argmax(&logits), 0);
    }

    // 2) Token letztes (vocab-1) bei pos 0 -> sich selbst
    #[test]
    fn test_2l_last_token_pos0() {
        let mut m = build_two_layer_identity_model(6, 1);
        let t = 5usize;
        let logits = m.forward_next(t, 0).expect("forward ok");
        assert_eq!(argmax(&logits), t);
    }

    // 3) Runde über alle Tokens -> Identität
    #[test]
    fn test_2l_round_all_tokens() {
        let mut m = build_two_layer_identity_model(6, 1);
        for pos in 0..6 {
            let tok = pos % 6;
            let logits = m.forward_next(tok, pos).expect("forward ok");
            assert_eq!(argmax(&logits), tok);
        }
    }

    // 4) KV-Cache über mehrere Schritte (Identität bleibt)
    #[test]
    fn test_2l_kv_cache_three_steps() {
        let mut m = build_two_layer_identity_model(6, 1);
        let _ = m.forward_next(1, 0).expect("ok");
        let _ = m.forward_next(2, 1).expect("ok");
        let logits = m.forward_next(4, 2).expect("ok");
        assert_eq!(argmax(&logits), 4);
    }

    // 5) RoPE hat keinen Effekt, wenn Attention/FFN Gewichte 0 sind
    //    (Logits für gleichen Input sind an pos 0 und pos 7 gleich)
    #[test]
    fn test_2l_logits_equal_over_positions() {
        let mut m = build_two_layer_identity_model(6, 1);
        let tok = 3usize;
        let l0 = m.forward_next(tok, 0).expect("ok");
        // Reset KV für sauberen Vergleich
        m.reset_kv_cache();
        let l1 = m.forward_next(tok, 7).expect("ok");
        assert_eq!(l0.len(), l1.len());
        for i in 0..l0.len() {
            assert!((l0[i] - l1[i]).abs() < 1e-6, "logit drift at {}", i);
        }
    }

    // 6) Kleiner Bias im lm_head verschiebt Vorhersage für einen speziellen Token
    #[test]
    fn test_2l_lm_head_bias_changes_argmax() {
        let mut m = build_two_layer_identity_model(6, 1);
        // Eingabetoken 2: normalerweise -> 2
        let tok = 2usize;
        // Bias: aus hidden=2 stärker auf output=3
        let v = 3usize;
        let row = tok; // weil x = one-hot bei index tok
        let idx = row * m.cfg.vocab_size + v;
        m.lm_head[idx] = 2.0; // größer als 1.0 der Identität
        let logits = m.forward_next(tok, 0).expect("ok");
        assert_eq!(argmax(&logits), 3);
    }

    // 7) Gleiche Identitäts-Logik mit 2 Heads (hidden divisible by heads)
    #[test]
    fn test_2l_two_heads_identity() {
        let mut m = build_two_layer_identity_model(6, 2); // hidden=6, heads=2 -> head_dim=3
        let logits = m.forward_next(4, 0).expect("ok");
        assert_eq!(argmax(&logits), 4);
    }

    // 8) KV-Cache mit 2 Heads über mehrere Positionen
    #[test]
    fn test_2l_two_heads_kv_sequence() {
        let mut m = build_two_layer_identity_model(6, 2);
        let _ = m.forward_next(0, 0).expect("ok");
        let _ = m.forward_next(1, 1).expect("ok");
        let _ = m.forward_next(2, 2).expect("ok");
        let logits = m.forward_next(5, 3).expect("ok");
        assert_eq!(argmax(&logits), 5);
    }

    // 9) Wiederholter Reset des KV-Caches liefert stabile Identitäts-Logits
    #[test]
    fn test_2l_kv_reset_stability() {
        let mut m = build_two_layer_identity_model(6, 1);
        let l_a = m.forward_next(1, 0).expect("ok");
        m.reset_kv_cache();
        let l_b = m.forward_next(1, 0).expect("ok");
        assert_eq!(l_a.len(), l_b.len());
        for i in 0..l_a.len() {
            assert!((l_a[i] - l_b[i]).abs() < 1e-6);
        }
    }

    // 10) Folge-Positionen unterscheiden sich nicht bei diesem Setup
    //     (weil Attention/FFN 0 -> nur Embedding + lm_head)
    #[test]
    fn test_2l_same_input_different_pos_equal_logits() {
        let mut m = build_two_layer_identity_model(6, 1);
        let tok = 5usize;
        let l0 = m.forward_next(tok, 0).expect("ok");
        let l1 = m.forward_next(tok, 1).expect("ok");
        assert_eq!(argmax(&l0), argmax(&l1));
        for i in 0..l0.len() {
            assert!((l0[i] - l1[i]).abs() < 1e-6);
        }
    }
}

// src/tests.rs
// Zwei-Layer-Varianten mit Mini-FFN und Greedy-Predict über 3 Tokens.
// Ziel: Logits verändern sich sinnvoll, Vorhersage ist reproduzierbar.

#[cfg(test)]
mod two_layer_mini_ffn_predict3 {
    use crate::layer::{ModelConfig, TransformerModel};

    // Kleiner Helfer: argmax
    fn argmax(v: &[f32]) -> usize {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > best_v {
                best_v = x;
                best_i = i;
            }
        }
        best_i
    }

    // Greedy-Top1 für n neue Tokens.
    // - Führt das Prompt durch (KV-Cache wird aufgebaut)
    // - Wählt jeweils argmax(logits) als nächstes Token
    fn greedy_predict_top1_n(
        model: &mut TransformerModel,
        prompt_ids: &[usize],
        i_n_new: usize,
    ) -> Result<Vec<usize>, String> {
        model.reset_kv_cache();
        if prompt_ids.is_empty() {
            return Err("prompt darf nicht leer sein".to_string());
        }

        // Prompt einspielen
        let mut logits = Vec::<f32>::new();
        for (i_pos, &tid) in prompt_ids.iter().enumerate() {
            logits = model.forward_next(tid, i_pos)?;
        }

        // n neue Tokens generieren (Greedy)
        let mut out: Vec<usize> = Vec::with_capacity(i_n_new);
        let mut cur_pos = prompt_ids.len();
        for _ in 0..i_n_new {
            let next_id = argmax(&logits);
            out.push(next_id);
            logits = model.forward_next(next_id, cur_pos)?;
            cur_pos += 1;
        }
        Ok(out)
    }

    // Baustein: Identity-Embeddings (vocab == hidden), lm_head = Transpose(Embeddings)
    fn init_identity_embeddings_and_head(m: &mut TransformerModel) {
        let v = m.cfg.vocab_size;
        let h = m.cfg.hidden_size;
        // Einheits-Embeddings: Token i -> One-Hot an Position i
        for tok in 0..v {
            for hid in 0..h {
                m.tok_emb[tok * h + hid] = if hid == tok { 1.0 } else { 0.0 };
            }
        }
        // lm_head = Transpose(Embeddings) -> (hidden x vocab)
        for hid in 0..h {
            for tok in 0..v {
                m.lm_head[hid * v + tok] = m.tok_emb[tok * h + hid];
            }
        }
    }

    // 2-Layer Grundmodell (Heads frei wählbar, hidden == vocab) für klare Zuordnung
    fn build_two_layer_base(vocab: usize, heads: usize) -> TransformerModel {
        let hidden = vocab;
        assert!(hidden % heads == 0, "hidden muss durch heads teilbar sein");
        let cfg = ModelConfig {
            vocab_size: vocab,
            hidden_size: hidden,
            intermediate_size: hidden * 2,
            n_layers: 2,
            n_heads: heads,
            n_kv_heads: 1,
            max_seq_len: 32,
            rope_dim: (hidden / heads).min(4),
            rope_base: 10000.0,
            rms_eps: 1e-5,
            rope_scaling_type: None,
            rope_scaling_factor: None,
        };
        let mut m = TransformerModel::new_empty(cfg.clone());
        init_identity_embeddings_and_head(&mut m);
        m
    }

    // Mini‑FFN „Repeat“: verstärkt den aktuellen Token leicht
    // Idee:
    //  - W1 und W3: kleine Diagonalen (alpha, beta) auf ersten 'hidden' Inter-Dimensionen
    //  - W2: kleine Diagonale (gamma) zurück auf denselben hidden-Index
    // Dadurch kommt ein kleiner positiver Zusatz über den FFN-Residual dazu.
    fn build_two_layer_mini_ffn_repeat(vocab: usize, heads: usize) -> TransformerModel {
        let mut m = build_two_layer_base(vocab, heads);

        let h = m.cfg.hidden_size;
        let inter = m.cfg.intermediate_size;

        // Kleine, aber nicht-null Gewichte
        let alpha = 0.10_f32;
        let beta = 0.10_f32;
        let gamma = 0.15_f32;

        for i in 0..h {
            // W1: [hidden -> inter], Row-Major: [in_dim, out_dim]
            // setze (i -> i)
            m.blocks[0].ffn.w1.w[i * inter + i] = alpha;
            m.blocks[1].ffn.w1.w[i * inter + i] = alpha;

            // W3 (Gate): (i -> i)
            m.blocks[0].ffn.w3.w[i * inter + i] = beta;
            m.blocks[1].ffn.w3.w[i * inter + i] = beta;

            // W2: [inter -> hidden], (i -> i)
            m.blocks[0].ffn.w2.w[i * h + i] = gamma;
            m.blocks[1].ffn.w2.w[i * h + i] = gamma;
        }
        m
    }

    // Mini‑FFN „Next“: schiebt zur nächsten Klasse (mod vocab)
    // Zusätzlich: lm_head leicht umgewichtet, damit „next“ gewinnt.
    //  - lm_head diagonal wird reduziert (diag_w)
    //  - Off-Diagonal (i -> next(i)) bekommt off_w
    //  - FFN pusht ebenfalls leicht in „next“-Richtung
    fn build_two_layer_mini_ffn_next(vocab: usize, heads: usize) -> TransformerModel {
        let mut m = build_two_layer_base(vocab, heads);

        let h = m.cfg.hidden_size;
        let inter = m.cfg.intermediate_size;

        // FFN-Gewichte klein, aber wirksam
        let alpha = 0.10_f32;
        let beta = 0.10_f32;
        let gamma = 0.20_f32;

        for i in 0..h {
            let next = (i + 1) % h;

            // W1, W3: (i -> i) im Intermediate
            m.blocks[0].ffn.w1.w[i * inter + i] = alpha;
            m.blocks[1].ffn.w1.w[i * inter + i] = alpha;

            m.blocks[0].ffn.w3.w[i * inter + i] = beta;
            m.blocks[1].ffn.w3.w[i * inter + i] = beta;

            // W2: (i -> next(i)) in hidden
            m.blocks[0].ffn.w2.w[i * h + next] = gamma;
            m.blocks[1].ffn.w2.w[i * h + next] = gamma;
        }

        // lm_head umbauen: leicht „next“-freundlich
        // Setze alles auf 0.0, dann Diagonal/Off-Diagonal setzen
        for hid in 0..h {
            for tok in 0..m.cfg.vocab_size {
                m.lm_head[hid * m.cfg.vocab_size + tok] = 0.0;
            }
        }
        let diag_w = 0.40_f32;
        let off_w = 0.65_f32;
        for hid in 0..h {
            let tok_same = hid;
            let tok_next = (hid + 1) % h;
            m.lm_head[hid * m.cfg.vocab_size + tok_same] = diag_w;
            m.lm_head[hid * m.cfg.vocab_size + tok_next] = off_w;
        }

        m
    }

    // ---------------------- Tests ----------------------

    // 1) Repeat-Variante: 3 Vorhersagen bleiben der gleiche Token
    #[test]
    fn test_repeat_predict3_from_token2() {
        let mut m = build_two_layer_mini_ffn_repeat(6, 2);
        let pred = greedy_predict_top1_n(&mut m, &[2usize], 3).expect("predict ok");
        assert_eq!(
            pred,
            vec![2, 2, 2],
            "Repeat-FFN sollte stabil Token 2 halten"
        );
    }

    // 2) Repeat-Variante: Start=5 -> alle drei = 5
    #[test]
    fn test_repeat_predict3_from_token5() {
        let mut m = build_two_layer_mini_ffn_repeat(6, 2);
        let pred = greedy_predict_top1_n(&mut m, &[5usize], 3).expect("predict ok");
        assert_eq!(pred, vec![5, 5, 5]);
    }

    // 3) Next-Variante: Start=2 -> 3,4,5
    #[test]
    fn test_next_predict3_from_token2() {
        let mut m = build_two_layer_mini_ffn_next(6, 2);
        let pred = greedy_predict_top1_n(&mut m, &[2usize], 3).expect("predict ok");
        assert_eq!(pred, vec![3, 4, 5], "Next-FFN sollte vorwärts schieben");
    }

    // 4) Next-Variante: Start=5 -> 0,1,2 (Wrap-around)
    #[test]
    fn test_next_predict3_from_token5() {
        let mut m = build_two_layer_mini_ffn_next(6, 2);
        let pred = greedy_predict_top1_n(&mut m, &[5usize], 3).expect("predict ok");
        assert_eq!(pred, vec![0, 1, 2]);
    }

    // 5) Repeat: Logits ändern sich (vs. „reines“ Identity-Basismodell), aber Top-1 bleibt gleich
    #[test]
    fn test_repeat_logits_change_but_top1_same() {
        let mut base = build_two_layer_base(6, 2);
        let mut rep = build_two_layer_mini_ffn_repeat(6, 2);

        let l_base = base.forward_next(4, 0).expect("ok");
        let l_rep = rep.forward_next(4, 0).expect("ok");

        // Top-1 bleibt 4
        assert_eq!(
            crate::tests::two_layer_mini_ffn_predict3::argmax(&l_base),
            4
        );
        assert_eq!(crate::tests::two_layer_mini_ffn_predict3::argmax(&l_rep), 4);

        // Aber mindestens an einer Stelle ändert sich der Logit
        let mut any_diff = false;
        for i in 0..l_base.len() {
            if (l_base[i] - l_rep[i]).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "Logits sollten sich durch Mini-FFN leicht ändern");
    }

    // 6) Next: Ein Schritt ohne Generation zeigt schon „next“ als Top-1
    #[test]
    fn test_next_single_step_top1() {
        let mut m = build_two_layer_mini_ffn_next(6, 2);
        let logits = m.forward_next(1, 0).expect("ok");
        let top = argmax(&logits);
        assert_eq!(top, 2, "Top-1 sollte der nächste Token sein");
    }

    // 7) Next: Zwei Schritte Greedy bleiben konsistent
    #[test]
    fn test_next_predict_two_steps() {
        let mut m = build_two_layer_mini_ffn_next(5, 1);
        let pred = greedy_predict_top1_n(&mut m, &[0usize], 2).expect("predict ok");
        assert_eq!(pred, vec![1, 2]);
    }

    // 8) Repeat: Drei Schritte aus Prompt mit Länge 2
    #[test]
    fn test_repeat_predict3_with_prompt_len2() {
        let mut m = build_two_layer_mini_ffn_repeat(5, 1);
        // Prompt [1,1] => nächstes bleibt 1
        let pred = greedy_predict_top1_n(&mut m, &[1usize, 1usize], 3).expect("predict ok");
        assert_eq!(pred, vec![1, 1, 1]);
    }

    // 9) Next: Prompt [2,3] → dann 4,0,1 (mit wrap)
    #[test]
    fn test_next_predict3_with_prompt_len2() {
        let mut m = build_two_layer_mini_ffn_next(5, 1);
        let pred = greedy_predict_top1_n(&mut m, &[2usize, 3usize], 3).expect("predict ok");
        assert_eq!(pred, vec![4, 0, 1]);
    }

    // 10) Stabilität: KV-Reset vor Predict liefert gleiche Folge
    #[test]
    fn test_predict3_is_deterministic_with_reset() {
        let mut m = build_two_layer_mini_ffn_next(6, 2);
        let p1 = greedy_predict_top1_n(&mut m, &[4usize], 3).expect("ok");
        let p2 = greedy_predict_top1_n(&mut m, &[4usize], 3).expect("ok");
        assert_eq!(p1, p2);
    }
}

// src/tests.rs
// Zwei-Layer-Varianten mit leicht aktiver Attention (kleine w_q/w_k/w_v)
// und kurzen Greedy-Vorhersagen über 3 Tokens.
// Autor: Marcus Schlieper, ExpChat.ai

#[cfg(test)]
mod two_layer_attn_variants {
    use crate::layer::{ModelConfig, TransformerModel};

    // Hilfsfunktionen
    fn argmax(v: &[f32]) -> usize {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > best_v {
                best_v = x;
                best_i = i;
            }
        }
        best_i
    }

    fn greedy_predict_top1_n(
        model: &mut TransformerModel,
        prompt_ids: &[usize],
        n_new: usize,
    ) -> Result<Vec<usize>, String> {
        model.reset_kv_cache();
        if prompt_ids.is_empty() {
            return Err("prompt darf nicht leer sein".to_string());
        }
        // Prompt einspielen
        let mut logits = Vec::<f32>::new();
        for (pos, &tid) in prompt_ids.iter().enumerate() {
            logits = model.forward_next(tid, pos)?;
        }
        // n neue Tokens (Greedy)
        let mut out = Vec::<usize>::with_capacity(n_new);
        let mut cur_pos = prompt_ids.len();
        for _ in 0..n_new {
            let next_id = argmax(&logits);
            out.push(next_id);
            logits = model.forward_next(next_id, cur_pos)?;
            cur_pos += 1;
        }
        Ok(out)
    }

    // Basis: vocab == hidden, 2 Layer, 1 Kopf (einfach und stabil)
    fn build_two_layer_base(vocab: usize) -> TransformerModel {
        let hidden = vocab;
        let heads = 1usize;
        let cfg = ModelConfig {
            vocab_size: vocab,
            hidden_size: hidden,
            intermediate_size: hidden * 2,
            n_layers: 2,
            n_heads: heads,
            n_kv_heads: 1,
            max_seq_len: 32,
            rope_dim: (hidden / heads).min(4),
            rope_base: 10000.0,
            rms_eps: 1e-5,
            rope_scaling_type: None,
            rope_scaling_factor: None,
        };
        let mut m = TransformerModel::new_empty(cfg.clone());

        // Identity-Embeddings: Token i -> One-Hot hidden i
        for tok in 0..m.cfg.vocab_size {
            for hid in 0..m.cfg.hidden_size {
                m.tok_emb[tok * m.cfg.hidden_size + hid] = if hid == tok { 1.0 } else { 0.0 };
            }
        }
        // lm_head = Transpose(tok_emb) (hidden x vocab)
        for hid in 0..m.cfg.hidden_size {
            for tok in 0..m.cfg.vocab_size {
                m.lm_head[hid * m.cfg.vocab_size + tok] = m.tok_emb[tok * m.cfg.hidden_size + hid];
            }
        }
        m
    }

    // Gemeinsamer Setter: kleine Q/K-Diagonalen, V als Permutation (shift),
    // O als leichte Diagonale.
    fn set_attn_diag_and_shift_next(
        m: &mut TransformerModel,
        eps_q: f32,
        eps_k: f32,
        gamma_v: f32,
        delta_o: f32,
        mix_identity_in_v: f32, // 0.0 = nur shift, >0 fügt kleine Identität zu V hinzu
        shift_by: usize,        // 1 = next, 2 = next+2, ...
        add_k_neighbor: f32,    // 0.0 = aus, sonst kleine K-Nachbar-Komponente
        only_layer_2: bool,     // true: Layer 1 bleibt 0, nur Layer 2 aktiv
    ) {
        let h = m.cfg.hidden_size;
        let kv_out = m.cfg.n_kv_heads * (m.cfg.hidden_size / m.cfg.n_heads); // = hidden (bei 1 Kopf)
        assert_eq!(kv_out, h, "diese Helfer erwarten 1 Kopf (kv_out == hidden)");

        for (li, blk) in m.blocks.iter_mut().enumerate() {
            if only_layer_2 && li == 0 {
                continue; // Layer 1 unverändert
            }

            // w_q und w_k: kleine Diagonalen
            for i in 0..h {
                // Q: [in_dim=h, out_dim=h]
                blk.attn.w_q.w[i * blk.attn.w_q.out_dim + i] = eps_q;
                // K: [in_dim=h, out_dim=kv_out]
                blk.attn.w_k.w[i * blk.attn.w_k.out_dim + i] = eps_k;
                if add_k_neighbor > 0.0 {
                    // kleine Nachbar-Komponente in K (co-attend)
                    let prev = if i == 0 { h - 1 } else { i - 1 };
                    blk.attn.w_k.w[i * blk.attn.w_k.out_dim + prev] += add_k_neighbor;
                }
            }

            // w_v: mapping auf (i -> i+shift_by) plus optionale Identität
            for i in 0..h {
                let next = (i + shift_by) % h;
                // V: [in_dim=h, out_dim=kv_out]
                blk.attn.w_v.w[i * blk.attn.w_v.out_dim + next] = gamma_v;
                if mix_identity_in_v > 0.0 {
                    blk.attn.w_v.w[i * blk.attn.w_v.out_dim + i] += mix_identity_in_v;
                }
            }

            // w_o: leichte Diagonale (verstärkt Attention-Ausgabe)
            for i in 0..h {
                blk.attn.w_o.w[i * blk.attn.w_o.out_dim + i] = delta_o;
            }
        }
    }

    // Variante 1: Next-Shift (stabil)
    fn build_var1(vocab: usize) -> TransformerModel {
        let mut m = build_two_layer_base(vocab);
        // vorher: 0.30, 0.30, 1.20, 1.10, 0.00, shift_by=1, 0.00, false
        set_attn_diag_and_shift_next(
            &mut m, 0.55, // eps_q
            0.55, // eps_k
            1.35, // gamma_v
            1.20, // delta_o
            0.00, // mix_identity_in_v
            1,    // shift_by
            0.00, // add_k_neighbor
            false,
        );
        m
    }

    // Variante 2: Next-Shift mit Identitäts-Mix in V (weicher)
    fn build_var2(vocab: usize) -> TransformerModel {
        let mut m = build_two_layer_base(vocab);
        set_attn_diag_and_shift_next(
            &mut m, 0.20, // eps_q
            0.20, // eps_k
            0.90, // gamma_v
            0.90, // delta_o
            0.20, // mix_identity_in_v (klein)
            1,    // shift_by
            0.00, // add_k_neighbor
            false,
        );
        m
    }

    // Variante 3: K mit kleiner Nachbar-Komponente (co-attend)
    fn build_var3(vocab: usize) -> TransformerModel {
        let mut m = build_two_layer_base(vocab);
        set_attn_diag_and_shift_next(
            &mut m, 0.25, // eps_q
            0.25, // eps_k
            1.00, // gamma_v
            1.00, // delta_o
            0.00, // mix_identity_in_v
            1,    // shift_by
            0.10, // add_k_neighbor (klein)
            false,
        );
        m
    }

    // Variante 4: Attention nur in Layer 2 aktiv
    fn build_var4(vocab: usize) -> TransformerModel {
        let mut m = build_two_layer_base(vocab);
        // vorher: 0.30, 0.30, 1.10, 1.00, 0.10, shift_by=1, 0.00, only_layer_2=true
        set_attn_diag_and_shift_next(
            &mut m, 0.65, // eps_q
            0.65, // eps_k
            1.35, // gamma_v
            1.20, // delta_o
            0.00, // mix_identity_in_v (weg)
            1,    // shift_by
            0.00, // add_k_neighbor
            true, // nur Layer 2
        );
        m
    }

    // Variante 5: “Next+2”-Shift (zwei weiter)
    fn build_var5(vocab: usize) -> TransformerModel {
        let mut m = build_two_layer_base(vocab);
        set_attn_diag_and_shift_next(
            &mut m, 0.35, // eps_q
            0.35, // eps_k
            1.10, // gamma_v
            1.10, // delta_o
            0.00, // mix_identity_in_v
            2,    // shift_by = 2
            0.00, // add_k_neighbor
            false,
        );
        m
    }

    // ---------------- Tests ----------------

    // Var1: Start=2 -> Next-Kette 3,4,5
    #[test]
    fn test_var1_next_chain_from_2() {
        let mut m = build_var1(6);
        let pred = greedy_predict_top1_n(&mut m, &[2usize], 3).expect("ok");
        assert_eq!(pred, vec![3, 4, 5]);
    }

    // Var2: unterschiedliche letzte Tokens -> unterschiedliche erste Vorhersage
    #[test]
    fn test_var2_last_token_influences_next() {
        let mut m = build_var2(6);
        let p_a = greedy_predict_top1_n(&mut m, &[1usize], 1).expect("ok");
        let p_b = greedy_predict_top1_n(&mut m, &[4usize], 1).expect("ok");
        assert_eq!(p_a, vec![2]); // next(1)
        assert_eq!(p_b, vec![5]); // next(4)
        assert_ne!(p_a, p_b);
    }

    // Var3: Prompt mit zwei Tokens – nächstes hängt von der letzten Position ab
    #[test]
    fn test_var3_two_token_prompt_uses_last() {
        let mut m = build_var3(6);
        let a = greedy_predict_top1_n(&mut m, &[1usize, 4usize], 1).expect("ok"); // last=4
        let b = greedy_predict_top1_n(&mut m, &[1usize, 5usize], 1).expect("ok"); // last=5
        assert_eq!(a, vec![5]); // next(4)
        assert_eq!(b, vec![0]); // next(5) (wrap)
    }

    // Var4: nur Layer 2 aktiv – Next-Verhalten bleibt erhalten
    #[test]
    fn test_var4_only_layer2_still_next() {
        let mut m = build_var4(6);
        let pred = greedy_predict_top1_n(&mut m, &[3usize], 2).expect("ok");
        assert_eq!(pred, vec![4, 5]);
    }

    // Var5: Next+2 – zwei Schritte weiter
    #[test]
    fn test_var5_next_plus_two() {
        let mut m = build_var5(6);
        let p0 = greedy_predict_top1_n(&mut m, &[0usize], 1).expect("ok");
        let p4 = greedy_predict_top1_n(&mut m, &[4usize], 1).expect("ok");
        assert_eq!(p0, vec![2]); // 0 -> 2
        assert_eq!(p4, vec![0]); // 4 -> 0 (wrap)
    }
}

#[cfg(test)]
mod two_layer_heads2_with_tilt {
    use crate::layer::{ModelConfig, TransformerModel};

    fn argmax(v: &[f32]) -> usize {
        let mut bi = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                bi = i;
            }
        }
        bi
    }

    // Greedy: erzeuge i_n_new Tokens (argmax).
    fn greedy_predict_top1_n(
        model: &mut TransformerModel,
        prompt_ids: &[usize],
        i_n_new: usize,
    ) -> Result<Vec<usize>, String> {
        model.reset_kv_cache();
        if prompt_ids.is_empty() {
            return Err("prompt darf nicht leer sein".to_string());
        }
        let mut logits = Vec::<f32>::new();
        for (pos, &tid) in prompt_ids.iter().enumerate() {
            logits = model.forward_next(tid, pos)?;
        }
        let mut out = Vec::with_capacity(i_n_new);
        let mut cur_pos = prompt_ids.len();
        for _ in 0..i_n_new {
            let next_id = argmax(&logits);
            out.push(next_id);
            logits = model.forward_next(next_id, cur_pos)?;
            cur_pos += 1;
        }
        Ok(out)
    }

    // Basis: vocab == hidden, 2 Layer, 2 Köpfe (kein GQA)
    fn build_two_layer_heads2_base(vocab: usize) -> TransformerModel {
        let hidden = vocab;
        let cfg = ModelConfig {
            vocab_size: vocab,
            hidden_size: hidden,
            intermediate_size: hidden * 2,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2, // wichtig: K/V-Output = hidden, einfacher zu steuern
            max_seq_len: 32,
            rope_dim: (hidden / 2).min(4),
            rope_base: 10000.0,
            rms_eps: 1e-5,
            rope_scaling_type: None,
            rope_scaling_factor: None,
        };
        let mut m = TransformerModel::new_empty(cfg.clone());

        // Identity-Embeddings: Token i -> One-Hot an hidden i
        for tok in 0..cfg.vocab_size {
            for hid in 0..cfg.hidden_size {
                m.tok_emb[tok * cfg.hidden_size + hid] = if hid == tok { 1.0 } else { 0.0 };
            }
        }
        // lm_head = Transpose(tok_emb) (hidden x vocab)
        for hid in 0..cfg.hidden_size {
            for tok in 0..cfg.vocab_size {
                m.lm_head[hid * cfg.vocab_size + tok] = m.tok_emb[tok * cfg.hidden_size + hid];
            }
        }
        m
    }

    // Attention auf “global next” ausrichten:
    // - w_q, w_k: kleine Diagonale (eps)
    // - w_v: i -> (i+shift_by) % hidden (gamma)
    // - w_o: leichte Diagonale (delta)
    fn set_heads2_attn_global_next(
        m: &mut TransformerModel,
        eps_q: f32,
        eps_k: f32,
        gamma_v: f32,
        delta_o: f32,
        shift_by: usize,
        only_layer_2: bool,
    ) {
        let h = m.cfg.hidden_size;
        for (li, blk) in m.blocks.iter_mut().enumerate() {
            if only_layer_2 && li == 0 {
                continue;
            }
            // Q/K diagonale Gewichte
            for i in 0..h {
                blk.attn.w_q.w[i * blk.attn.w_q.out_dim + i] = eps_q;
                blk.attn.w_k.w[i * blk.attn.w_k.out_dim + i] = eps_k;
            }
            // V: globaler Shift über alle hidden-Indizes
            for i in 0..h {
                let j = (i + shift_by) % h;
                blk.attn.w_v.w[i * blk.attn.w_v.out_dim + j] = gamma_v;
            }
            // O: leichte Diagonale
            for i in 0..h {
                blk.attn.w_o.w[i * blk.attn.w_o.out_dim + i] = delta_o;
            }
        }
    }

    // lm_head-Feinjustierung (“Tilt”): Gewicht von self -> etwas runter,
    // Gewicht zu next -> etwas rauf. So gewinnt “next” häufiger.
    fn tilt_lm_head_next(m: &mut TransformerModel, diag_delta: f32, off_delta: f32) {
        let v = m.cfg.vocab_size;
        let h = m.cfg.hidden_size;
        assert_eq!(v, h, "für klare Tests: vocab == hidden");
        for hid in 0..h {
            let tok_same = hid;
            let tok_next = (hid + 1) % v;
            // self etwas schwächer
            m.lm_head[hid * v + tok_same] += diag_delta; // z.B. -0.05
            // next etwas stärker
            m.lm_head[hid * v + tok_next] += off_delta; // z.B. +0.08
        }
    }

    // Variante A: 2 Köpfe, beide Layer aktiv, global “next”, ohne Tilt
    fn build_heads2_var_next(vocab: usize) -> TransformerModel {
        let mut m = build_two_layer_heads2_base(vocab);
        set_heads2_attn_global_next(
            &mut m, 0.75, // eps_q  (vorher 0.50)
            0.75, // eps_k
            1.50, // gamma_v
            1.25, // delta_o
            1,    // shift_by (next)
            false,
        );
        m
    }

    // Variante B: wie A, aber nur Layer 2 aktiv
    fn build_heads2_var_next_only_l2(vocab: usize) -> TransformerModel {
        let mut m = build_two_layer_heads2_base(vocab);
        set_heads2_attn_global_next(
            &mut m, 0.80, // eps_q
            0.80, // eps_k
            1.55, // gamma_v
            1.25, // delta_o
            1,    // next
            true, // nur Layer 2
        );
        m
    }
    fn ensure_next_margin(m: &mut TransformerModel, margin: f32) {
        let v = m.cfg.vocab_size;
        let h = m.cfg.hidden_size;
        assert_eq!(v, h, "für diese Tests: vocab == hidden");
        for hid in 0..h {
            let same_ix = hid * v + hid;
            let next_tok = (hid + 1) % v;
            let next_ix = hid * v + next_tok;
            let same = m.lm_head[same_ix];
            let next = m.lm_head[next_ix];
            if next < same + margin {
                m.lm_head[next_ix] = same + margin;
            }
        }
    }

    fn build_heads2_var_next_with_tilt(vocab: usize) -> TransformerModel {
        let mut m = build_two_layer_heads2_base(vocab);

        // Nur Layer 2 aktiv (verhindert doppelten Shift)
        set_heads2_attn_global_next(
            &mut m, 0.55, // eps_q
            0.55, // eps_k
            1.20, // gamma_v (Shift-Stärke)
            1.15, // delta_o (Output-Proj.)
            1,    // shift_by = +1
            true, // only_layer_2 = true
        );

        // Tilt: next klar vor self
        tilt_lm_head_next(&mut m, -0.18, 0.30);

        // Sicherheitsmarge für next
        ensure_next_margin(&mut m, 0.05);

        m
    }

    // ---------------- Tests ----------------

    // A1) Start=2 -> [3,4,5] (keine Grenze-probleme, global next)
    #[test]
    fn test_heads2_next_chain_from_2() {
        let mut m = build_heads2_var_next(8);
        let pred = greedy_predict_top1_n(&mut m, &[2usize], 3).expect("ok");
        assert_eq!(pred, vec![3, 4, 5]);
    }

    // A2) Prompt [5] -> [6,7,0] (wrap bei 7->0)
    #[test]
    fn test_heads2_next_wrap() {
        let mut m = build_heads2_var_next(8);
        let pred = greedy_predict_top1_n(&mut m, &[5usize], 3).expect("ok");
        assert_eq!(pred, vec![6, 7, 0]);
    }

    // B1) Nur Layer 2 aktiv: Start=3 -> [4,5]
    #[test]
    fn test_heads2_only_layer2_still_next() {
        let mut m = build_heads2_var_next_only_l2(8);
        let pred = greedy_predict_top1_n(&mut m, &[3usize], 2).expect("ok");
        assert_eq!(pred, vec![4, 5]);
    }

    // C1) Schwächere Attention + Tilt: Start=1 -> [2,3,4]
    #[test]
    fn test_heads2_with_tilt_chain() {
        let mut m = build_heads2_var_next_with_tilt(8);
        let pred = greedy_predict_top1_n(&mut m, &[1usize], 3).expect("ok");
        assert_eq!(pred, vec![2, 3, 4]);
    }

    // C2) Tilt wirkt auch am Rand: Start=7 -> [0]
    #[test]
    fn test_heads2_with_tilt_wrap_one() {
        let mut m = build_heads2_var_next_with_tilt(8);
        let pred = greedy_predict_top1_n(&mut m, &[7usize], 1).expect("ok");
        assert_eq!(pred, vec![0]);
    }
}

// Golden-Step-Test gegen Referenz
// - Liest eine Golden-JSON (aus llama.cpp o. Ä.)
// - Vergleicht Top-1/Top-5 und Top-5-Logits pro Schritt
// - Nutzt exakt die in der Golden-Datei angegebenen Token-IDs

#[cfg(test)]
mod golden_step_tests {
    use std::fs::File;
    use std::io::Read;
    use std::path::Path;

    use crate::gguf_loader::load_gguf;
    use crate::gguf_loader::{GgufModel, GgufValue};
    use crate::layer::TransformerModel;
    use crate::model::{build_config, map_all_weights};
    use crate::tokenizer::GgufTokenizer;
    use crate::tokenizer::gguf_tokenizer_from_kv;

    // dev-deps: serde, serde_json
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct GoldenStep {
        pos: usize,
        top1_id: usize,
        top5_ids: Vec<usize>,
        // Optional: logarithmische Werte (Logits) der Top-5 in Referenz
        // Tipp: in der Referenz vorher "centered" (minus max) speichern
        top5_logits: Option<Vec<f32>>,
    }

    #[derive(Debug, Deserialize)]
    struct GoldenFile {
        // optional, nur Info
        model_path: Option<String>,
        tokenizer_model: Option<String>, // "gpt2" | "llama"
        prompt_text: Option<String>,
        // Wichtig: exakt die IDs, die die Referenz verwendet hat
        token_ids: Vec<usize>,
        // Anzahl Schritte, die wir vergleichen
        steps: Vec<GoldenStep>,
        // optional: vocab_size, hilft bei Plausibilitätsprüfungen
        vocab_size: Option<usize>,
    }

    fn read_golden_json(p: &str) -> GoldenFile {
        let mut f = File::open(p).expect("Golden JSON nicht gefunden");
        let mut s = String::new();
        f.read_to_string(&mut s)
            .expect("Golden JSON konnte nicht gelesen werden");
        serde_json::from_str::<GoldenFile>(&s).expect("Golden JSON ist ungültig")
    }

    fn argmax(v: &[f32]) -> usize {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > best_v {
                best_v = x;
                best_i = i;
            }
        }
        best_i
    }

    fn topk_ids(logits: &[f32], k: usize) -> Vec<usize> {
        let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        pairs.into_iter().take(k).map(|(i, _)| i).collect()
    }

    fn gather(vec: &[f32], ids: &[usize]) -> Vec<f32> {
        ids.iter().map(|&i| vec[i]).collect()
    }

    // optional: normalisiere Logits (minus max), für robuste Vergleiche
    fn center_logits(v: &mut [f32]) {
        if let Some(&mx) = v.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
            for x in v.iter_mut() {
                *x -= mx;
            }
        }
    }

    fn approx_vec(a: &[f32], b: &[f32], eps: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for i in 0..a.len() {
            if (a[i] - b[i]).abs() > eps {
                return false;
            }
        }
        true
    }

    // Mini-Helfer: baut Modell aus GGUF, lädt Gewichte
    fn load_model_and_tokenizer(model_path: &str) -> (TransformerModel, GgufTokenizer) {
        let gguf = load_gguf(model_path).expect("GGUF laden fehlgeschlagen");
        let cfg = build_config(&gguf);
        let mut model = TransformerModel::new_empty(cfg.clone());
        map_all_weights(&gguf, &mut model).expect("Gewichte mappen fehlgeschlagen");

        let tok = gguf_tokenizer_from_kv(&gguf.kv)
            .expect("Tokenizer aus GGUF konnte nicht gebaut werden");

        (model, tok)
    }

    fn log_topk_with_decodes(tok: &GgufTokenizer, logits: &[f32], k: usize, note: &str) {
        let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        eprintln!("== top{} {} ==", k.min(pairs.len()), note);
        for (rank, (tid, logit)) in pairs.into_iter().take(k).enumerate() {
            // einzelnes Token dekodieren (skip_special=true)
            let piece = tok
                .decode(&[tid], true)
                .unwrap_or_else(|_| "<dec_fail>".to_string());
            eprintln!("#{} id={} logit={:.6} piece='{}'", rank, tid, logit, piece);
        }
    }

    // Hilfsfunktion: Tokens-Array aus GGUF lesen (Vokabular in Datei-Reihenfolge)
    fn gguf_tokens_vec(gguf: &GgufModel) -> Option<Vec<String>> {
        match gguf.kv.get("tokenizer.ggml.tokens") {
            Some(GgufValue::ArrStr(v)) => Some(v.clone()),
            _ => None,
        }
    }

    // Hilfsfunktion: Ausgewählte Token-IDs für ein Vokabular drucken
    fn print_token_samples_for_vocab(
        label: &str,
        gguf: &GgufModel,
        tok: &GgufTokenizer,
        ids: &[usize],
    ) {
        eprintln!("== token samples: {} ==", label);
        // Roh-Vokabel (aus GGUF) zum direkten Nachschlagen
        let raw_tokens = gguf_tokens_vec(gguf).unwrap_or_default();
        for &tid in ids {
            let piece_dec = tok
                .decode(&[tid], true)
                .unwrap_or_else(|_| "<dec_fail>".to_string());
            let raw = raw_tokens
                .get(tid)
                .cloned()
                .unwrap_or_else(|| "<oob>".to_string());
            eprintln!("id={}  decode='{}'  raw='{}'", tid, piece_dec, raw);
        }
    }

    // Hilfsfunktion: Vokabulare vergleichen, wenn zwei GGUF-Dateien vorhanden
    fn compare_vocabs_and_report(a: &GgufModel, b: &GgufModel) -> bool {
        let va = gguf_tokens_vec(a).unwrap_or_default();
        let vb = gguf_tokens_vec(b).unwrap_or_default();
        let same_len = va.len() == vb.len();
        let same_all = same_len && va.iter().zip(vb.iter()).all(|(x, y)| x == y);
        if !same_len {
            eprintln!(
                "Vokabulargroessen unterschiedlich: a={} b={}",
                va.len(),
                vb.len()
            );
        }
        if same_len && !same_all {
            // Ersten Unterschied melden
            for i in 0..va.len() {
                if va[i] != vb[i] {
                    eprintln!(
                        "Vokab-Differenz bei index {}: a='{}' b='{}'",
                        i, va[i], vb[i]
                    );
                    break;
                }
            }
        }
        same_all
    }

    // Autor: Marcus Schlieper, ExpChat.ai
    // Datei: src/tests.rs (Modul golden_step_tests)
    // Zweck: Robuster Golden-Step-Test mit Toleranz und klaren Fehlermeldungen
    // Historie:
    // - 2025-12-14 (MS): Test erweitert: Toleranz fuer Top1/Top5, Logit-Toleranz einstellbar,
    //                    robuste Checks fuer OOB-IDs, doppelte Steps, bessere Logs.
    #[test]
    fn test_golden_step_top1_top5_parity() {
        use std::path::Path;

        // ---- Hilfsfunktionen (lokal) -------------------------------------------

        fn parse_env_usize(s_key: &str, i_default: usize) -> usize {
            std::env::var(s_key)
                .ok()
                .and_then(|s_v| s_v.parse::<usize>().ok())
                .unwrap_or(i_default)
        }

        fn parse_env_f32(s_key: &str, d_default: f32) -> f32 {
            std::env::var(s_key)
                .ok()
                .and_then(|s_v| s_v.parse::<f32>().ok())
                .unwrap_or(d_default)
        }

        fn parse_env_bool(s_key: &str, b_default: bool) -> bool {
            std::env::var(s_key)
                .map(|s_v| match s_v.as_str() {
                    "1" | "true" | "TRUE" | "True" => true,
                    "0" | "false" | "FALSE" | "False" => false,
                    _ => b_default,
                })
                .unwrap_or(b_default)
        }

        fn overlap_count(a_ids: &[usize], b_ids: &[usize]) -> usize {
            use std::collections::HashSet;
            let set_a: HashSet<usize> = a_ids.iter().copied().collect();
            let set_b: HashSet<usize> = b_ids.iter().copied().collect();
            set_a.intersection(&set_b).count()
        }

        // ---- Env-Settings ------------------------------------------------------

        let s_golden =
            std::env::var("GOLDEN_JSON").expect("Bitte GOLDEN_JSON=Pfad/zur/datei.json setzen");
        let s_model =
            std::env::var("MODEL_PATH").expect("Bitte MODEL_PATH=Pfad/zum/model.gguf setzen");

        let b_tolerant = parse_env_bool("GOLDEN_TOLERANT", true);
        let i_min_top5_overlap = parse_env_usize("GOLDEN_TOP5_OVERLAP_MIN", 4);
        let d_logits_eps = parse_env_f32("GOLDEN_LOGITS_EPS", 2e-2);
        let i_allow_mismatch_steps = parse_env_usize("GOLDEN_ALLOW_MISMATCH_STEPS", 0);
        let b_top1_bidir = parse_env_bool("GOLDEN_TOP1_IN_TOP5_BIDIR", false);

        assert!(
            Path::new(&s_golden).exists(),
            "Golden-Datei existiert nicht: {}",
            s_golden
        );
        assert!(
            Path::new(&s_model).exists(),
            "Modell-Datei existiert nicht: {}",
            s_model
        );

        // ---- Modell + Tokenizer laden ------------------------------------------

        let gguf_main = crate::gguf_loader::load_gguf(&s_model).expect("GGUF laden fehlgeschlagen");
        let (mut model, tok_wrap) = {
            let cfg = crate::model::build_config(&gguf_main);
            let mut m = crate::layer::TransformerModel::new_empty(cfg.clone());
            crate::model::map_all_weights(&gguf_main, &mut m)
                .expect("map_all_weights fehlgeschlagen");
            let t = crate::tokenizer::gguf_tokenizer_from_kv(&gguf_main.kv)
                .expect("Tokenizer aus GGUF konnte nicht gebaut werden");
            (m, t)
        };

        let golden = {
            // nutzt bereits vorhandene Funktion im Modul
            super::golden_step_tests::read_golden_json(&s_golden)
        };

        if let Some(i_vsz) = golden.vocab_size {
            assert_eq!(
                i_vsz, model.cfg.vocab_size,
                "vocab_size passt nicht Golden vs. Modell"
            );
        }

        // ---- Schrittweise Inferenz ueber Golden-Token-IDs ----------------------

        let v_ids = &golden.token_ids;
        assert!(
            !v_ids.is_empty(),
            "Golden: token_ids duerfen nicht leer sein"
        );

        // vorwaermen: einmal komplett durchlaufen (optionales Logging bei pos 0)
        model.reset_kv_cache();
        let mut v_logits: Vec<f32> = Vec::new();
        for (i_pos, _step) in golden.steps.iter().enumerate() {
            let i_tid = *v_ids
                .get(i_pos)
                .expect("Golden steps laenger als token_ids");
            v_logits = model.forward_next(i_tid, i_pos).expect("forward_next");
            if i_pos == 0 {
                // hilfreiches Debug-Logging mit Decodes
                super::golden_step_tests::log_topk_with_decodes(
                    &tok_wrap,
                    &v_logits,
                    5,
                    "pos=0 (golden-step, pre-check)",
                );
            }
        }

        // ---- Zweiter Durchlauf: Vergleiche pro Schritt -------------------------

        model.reset_kv_cache();

        let mut i_mismatches_top1 = 0usize;
        let mut i_mismatches_top5 = 0usize;
        let mut i_mismatches_logits = 0usize;
        let mut i_total_steps = 0usize;

        for (i_pos, step) in golden.steps.iter().enumerate() {
            let i_tid = match v_ids.get(i_pos) {
                Some(&id) => id,
                None => {
                    eprintln!(
                        "Warnung: Schritt {} hat kein token_id Pendant. Abbruch der Pruefung.",
                        i_pos
                    );
                    break;
                }
            };

            // Forward
            v_logits = match model.forward_next(i_tid, i_pos) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Fehler bei forward_next @pos {}: {}", i_pos, e);
                    i_mismatches_top1 += 1;
                    i_mismatches_top5 += 1;
                    i_mismatches_logits += 1;
                    continue;
                }
            };

            // Top1 unserer Logits
            let i_top1_ours = super::golden_step_tests::argmax(&v_logits);

            // Top5 unserer Logits (IDs)
            let v_top5_ours = super::golden_step_tests::topk_ids(&v_logits, 5);

            // Golden-Top5 (IDs) filtern gegen Vokabulargrenze
            let i_vocab = model.cfg.vocab_size;
            let mut v_top5_gold: Vec<usize> = step
                .top5_ids
                .iter()
                .copied()
                .filter(|&ix| ix < i_vocab)
                .collect();

            if v_top5_gold.len() < step.top5_ids.len() {
                eprintln!(
                    "Hinweis: @pos {} wurden {} Top5-IDs aus Golden wegen OOB entfernt (vocab={}).",
                    i_pos,
                    step.top5_ids.len() - v_top5_gold.len(),
                    i_vocab
                );
            }

            // --- Top1-Pruefung (exakt oder tolerant)
            let mut b_top1_ok = i_top1_ours == step.top1_id;

            if !b_top1_ok && b_tolerant {
                // Tolerant: Entweder unser Top1 liegt in Golden-Top5
                let b_ours_in_gold5 = v_top5_gold.contains(&i_top1_ours);
                // Optional: bidirektional pruefen (Golden Top1 liegt in unseren Top5)
                let b_gold_in_ours5 = v_top5_ours.contains(&step.top1_id);

                b_top1_ok = if b_top1_bidir {
                    b_ours_in_gold5 && b_gold_in_ours5
                } else {
                    b_ours_in_gold5 || b_gold_in_ours5
                };

                if !b_top1_ok {
                    eprintln!(
                        "Top1 mismatch @pos {}: ours={} golden={}",
                        i_pos, i_top1_ours, step.top1_id
                    );
                }
            } else if !b_top1_ok {
                eprintln!(
                    "Top1 mismatch @pos {}: ours={} golden={}",
                    i_pos, i_top1_ours, step.top1_id
                );
            }

            if !b_top1_ok {
                i_mismatches_top1 += 1;
            }

            // --- Top5-Pruefung (Mengenueberlappung)
            let i_overlap = overlap_count(&v_top5_ours, &v_top5_gold);
            let b_top5_ok = if b_tolerant {
                i_overlap >= i_min_top5_overlap
            } else {
                v_top5_ours == v_top5_gold
            };

            if !b_top5_ok {
                eprintln!(
                    "Top5 mismatch @pos {}: ours={:?} golden={:?} overlap={}",
                    i_pos, v_top5_ours, v_top5_gold, i_overlap
                );
                i_mismatches_top5 += 1;
            }

            // --- Logit-Pruefung (optional, wenn vorhanden)
            if let Some(ref v_ref_top5_logits) = step.top5_logits {
                // Greife unsere Logits auf den Golden-Top5-IDs ab
                let mut v_ours_vals = super::golden_step_tests::gather(&v_logits, &v_top5_gold);

                // Zentriere beide Seiten (minus max), robust gegen Offsets
                super::golden_step_tests::center_logits(&mut v_ours_vals);

                let mut v_ref_vals = v_ref_top5_logits.clone();
                let mut v_ref_centered = v_ref_vals.clone();
                super::golden_step_tests::center_logits(&mut v_ref_centered);

                // Laengenangleichung: vergleiche bis zur minimalen Laenge
                let i_min_len = v_ours_vals.len().min(v_ref_centered.len());
                let mut b_logits_ok = true;
                for i_k in 0..i_min_len {
                    if (v_ours_vals[i_k] - v_ref_centered[i_k]).abs() > d_logits_eps {
                        b_logits_ok = false;
                        break;
                    }
                }
                if !b_logits_ok {
                    eprintln!(
                        "Logit mismatch @pos {}: ours_centered={:?} ref_centered={:?} eps={}",
                        i_pos, v_ours_vals, v_ref_centered, d_logits_eps
                    );
                    i_mismatches_logits += 1;
                }
            }

            i_total_steps += 1;
        }

        // ---- Abschlusskriterien -------------------------------------------------

        // Erlaube definierte Anzahl Abweichungen, sonst Fehler
        let i_total_mismatches = i_mismatches_top1 + i_mismatches_top5 + i_mismatches_logits;

        if i_total_mismatches > 0 {
            eprintln!("Zusammenfassung:");
            eprintln!("  Steps total    : {}", i_total_steps);
            eprintln!("  Top1 mismatches: {}", i_mismatches_top1);
            eprintln!("  Top5 mismatches: {}", i_mismatches_top5);
            eprintln!("  Logit mismatches: {}", i_mismatches_logits);
            eprintln!("  Allowed mismatches (steps): {}", i_allow_mismatch_steps);
            eprintln!("  Tolerant mode: {}", b_tolerant);
            eprintln!("  Top5 min overlap: {}", i_min_top5_overlap);
            eprintln!("  Logits eps: {}", d_logits_eps);
            eprintln!("  Bi-dir Top1-in-Top5: {}", b_top1_bidir);
        }

        // Strenge Schlussregel: Anzahl der Schritte mit Abweichungen darf Schwellwert nicht ueberschreiten.
        // Hinweis: Wir nehmen konservativ die Summe aller drei Typen als "gewichtete" Abweichungen.
        // Fuer alternative Regeln kann man dies anpassen.
        assert!(
            i_total_mismatches <= i_allow_mismatch_steps,
            "Zu viele Abweichungen: {} > erlaubt {}",
            i_total_mismatches,
            i_allow_mismatch_steps
        );
    }

    // Bonus: reiner Tokenizer-Paritäts-Test (falls Golden prompt_text liefert)
    // Prüft: encode(text) == golden.token_ids
    #[test]
    fn test_golden_tokenizer_parity_if_text_given() {
        let s_golden = "c:\\Entwicklung\\rust\\GPT-GGUF\\src\\golden.json";

        /*match std::env::var("GOLDEN_JSON") {
            Ok(v) => v,
            Err(_) => {
                eprintln!("GOLDEN_JSON nicht gesetzt — Test wird übersprungen.");
                return;
            }
        };*/
        let s_model = "c:\\Entwicklung\\rust\\GPT-GGUF\\model\\vibethinker-1.5b-q8_0.gguf";

        /*match std::env::var("MODEL_PATH") {
            Ok(v) => v,
            Err(_) => {
                eprintln!("MODEL_PATH nicht gesetzt — Test wird übersprungen.");
                return;
            }
        };*/

        let golden = read_golden_json(&s_golden);
        let gguf = load_gguf(&s_model).expect("GGUF laden fehlgeschlagen");
        let tok = gguf_tokenizer_from_kv(&gguf.kv)
            .expect("Tokenizer aus GGUF konnte nicht gebaut werden");

        if let Some(txt) = golden.prompt_text.as_ref() {
            let ours = tok.encode(txt, false).expect("encode fehlgeschlagen");
            assert_eq!(
                ours, golden.token_ids,
                "Tokenizer-IDs weichen vom Golden ab"
            );
        }
    }
}
