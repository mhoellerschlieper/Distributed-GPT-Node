// src/p2p_codec.rs
// ------------------------------------------------------------
// libp2p request-response codec fuer bincode payload
//
// Ziel
// - Request und Response sind Vec u8 (bincode payload)
// - laengenpraefix framing
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie:
// - 2025-12-26 Marcus Schlieper: initiale version
// - 2025-12-26 Marcus Schlieper: libp2p 0.56 api fixes (ProtocolName entfernt)
// ------------------------------------------------------------

use async_trait::async_trait;
use libp2p::futures;
use libp2p::request_response::Codec;
use std::io;

#[derive(Clone)]
pub struct BlockProto;

impl AsRef<str> for BlockProto {
    fn as_ref(&self) -> &str {
        "/expchat/block-run/1"
    }
}

#[derive(Clone, Default)]
pub struct BlockCodec;

#[async_trait]
impl Codec for BlockCodec {
    type Protocol = BlockProto;
    type Request = Vec<u8>;
    type Response = Vec<u8>;

    async fn read_request<T>(
        &mut self,
        _: &Self::Protocol,
        o_io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        read_len_prefixed(o_io, 8 * 1024 * 1024).await
    }

    async fn read_response<T>(
        &mut self,
        _: &Self::Protocol,
        o_io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        read_len_prefixed(o_io, 8 * 1024 * 1024).await
    }

    async fn write_request<T>(
        &mut self,
        _: &Self::Protocol,
        o_io: &mut T,
        v_data: Self::Request,
    ) -> io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        write_len_prefixed(o_io, &v_data).await
    }

    async fn write_response<T>(
        &mut self,
        _: &Self::Protocol,
        o_io: &mut T,
        v_data: Self::Response,
    ) -> io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        write_len_prefixed(o_io, &v_data).await
    }
}

async fn read_len_prefixed<T>(o_io: &mut T, i_max: usize) -> io::Result<Vec<u8>>
where
    T: futures::AsyncRead + Unpin + Send,
{
    use libp2p::futures::AsyncReadExt;

    let mut v_len = [0u8; 4];
    o_io.read_exact(&mut v_len).await?;
    let i_len = u32::from_be_bytes(v_len) as usize;

    if i_len > i_max {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "msg too large"));
    }

    let mut v_buf = vec![0u8; i_len];
    o_io.read_exact(&mut v_buf).await?;
    Ok(v_buf)
}

async fn write_len_prefixed<T>(o_io: &mut T, v_data: &[u8]) -> io::Result<()>
where
    T: futures::AsyncWrite + Unpin + Send,
{
    use libp2p::futures::AsyncWriteExt;

    let i_len = v_data.len();
    if i_len > (u32::MAX as usize) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "msg too large"));
    }

    o_io.write_all(&(i_len as u32).to_be_bytes()).await?;
    o_io.write_all(v_data).await?;
    o_io.flush().await?;
    Ok(())
}
