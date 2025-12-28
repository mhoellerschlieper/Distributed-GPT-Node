// src/p2p_node.rs
// ------------------------------------------------------------
// libp2p Node Glue: Swarm Task plus Request API
//
// Ziel
// - ein Task besitzt den Swarm
// - andere Teile senden Requests per mpsc
// - Antwort kommt per oneshot
// - eingehende Requests werden verarbeitet (server handler)
// - liste der verbundenen peers wird verwaltet und ausgegeben
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie
// - 2025-12-26 Marcus Schlieper: initiale version
// - 2025-12-26 Marcus Schlieper: swarm task architektur mit mpsc
// - 2025-12-28 Marcus Schlieper: server handling plus peers list
// ------------------------------------------------------------

use crate::p2p_codec::BlockProto;

use libp2p::{
    identify, mdns, noise, ping, request_response,
    swarm::{NetworkBehaviour, Swarm, SwarmEvent},
    tcp, yamux, Multiaddr, PeerId, Transport,
};

use libp2p::futures::StreamExt;

use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::{mpsc, oneshot, Mutex};

// ---------------- Types ----------------

pub type SwarmType = Swarm<P2pBehaviour>;
pub type RequestIdType = request_response::OutboundRequestId;

pub type ServerHandler = Arc<
    dyn Fn(
            PeerId,
            Vec<u8>,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>, String>> + Send>>
        + Send
        + Sync,
>;

const REQUEST_TIMEOUT_SEC: u64 = 10;

// ---------------- Behaviour ----------------

#[derive(Debug)]
pub enum SwarmControlMsg {
    RegisterPeerAddr { s_peer_id: String, s_addr: String },
    ConnectPeer { s_peer_id: String, s_addr: String },
}

fn parse_peer_id(s_peer_id: &str) -> Result<PeerId, String> {
    s_peer_id
        .trim()
        .parse::<PeerId>()
        .map_err(|_| "peer id ungueltig".to_string())
}

fn parse_multiaddr(s_addr: &str) -> Result<Multiaddr, String> {
    s_addr
        .trim()
        .parse::<Multiaddr>()
        .map_err(|_| "multiaddr ungueltig".to_string())
}

#[derive(NetworkBehaviour)]
pub struct P2pBehaviour {
    pub rr: request_response::Behaviour<crate::p2p_codec::BlockCodec>,
    pub mdns: mdns::tokio::Behaviour,
    pub identify: identify::Behaviour,
    pub ping: ping::Behaviour,
}

// ---------------- Runtime API ----------------

#[derive(Debug)]
pub struct OutboundCall {
    pub o_peer: PeerId,
    pub v_req: Vec<u8>,
    pub o_tx: oneshot::Sender<Result<Vec<u8>, String>>,
}

#[derive(Clone)]
pub struct P2pRuntime {
    pub o_tx: mpsc::Sender<OutboundCall>,
    pub o_self_peer_id: PeerId,
    pub o_connected_peers: Arc<Mutex<HashSet<PeerId>>>,
    pub o_ctl_tx: mpsc::Sender<SwarmControlMsg>,
}

pub fn build_runtime(
    i_listen_tcp_port: u16,
    o_key: libp2p::identity::Keypair,
) -> Result<
    (
        P2pRuntime,
        SwarmType,
        tokio::sync::mpsc::Receiver<OutboundCall>,
        tokio::sync::mpsc::Receiver<SwarmControlMsg>,
    ),
    String,
> {
    let o_peer_id = PeerId::from(o_key.public());

    let o_tcp = tcp::tokio::Transport::new(tcp::Config::default().nodelay(true));
    let o_noise = noise::Config::new(&o_key).map_err(|e| format!("noise: {}", e))?;
    let o_yamux = yamux::Config::default();

    let o_transport = o_tcp
        .upgrade(libp2p::core::upgrade::Version::V1)
        .authenticate(o_noise)
        .multiplex(o_yamux)
        .boxed();

    let mut o_rr_cfg = request_response::Config::default();
    o_rr_cfg.set_request_timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SEC));

    let o_rr = request_response::Behaviour::new(
        [(BlockProto, request_response::ProtocolSupport::Full)],
        o_rr_cfg,
    );

    let o_mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), o_peer_id)
        .map_err(|e| format!("mdns: {}", e))?;

    let o_identify = identify::Behaviour::new(identify::Config::new(
        "expchat-p2p/1".to_string(),
        o_key.public(),
    ));

    let o_ping = ping::Behaviour::new(ping::Config::new());

    let o_behaviour = P2pBehaviour {
        rr: o_rr,
        mdns: o_mdns,
        identify: o_identify,
        ping: o_ping,
    };

    let mut o_swarm = Swarm::new(
        o_transport,
        o_behaviour,
        o_peer_id,
        libp2p::swarm::Config::with_tokio_executor(),
    );

    let s_listen = format!("/ip4/0.0.0.0/tcp/{}", i_listen_tcp_port);
    let a_listen: Multiaddr = s_listen
        .parse()
        .map_err(|_| "listen addr parse fehler".to_string())?;

    Swarm::listen_on(&mut o_swarm, a_listen).map_err(|e| format!("listen_on: {}", e))?;

    let (o_tx, o_rx) = mpsc::channel::<OutboundCall>(64);

    // neu: control channel
    let (o_ctl_tx, o_ctl_rx) = mpsc::channel::<SwarmControlMsg>(16);

    let o_connected_peers: Arc<Mutex<HashSet<PeerId>>> = Arc::new(Mutex::new(HashSet::new()));

    Ok((
        P2pRuntime {
            o_tx,
            o_self_peer_id: o_peer_id,
            o_connected_peers: o_connected_peers.clone(),
            o_ctl_tx,
        },
        o_swarm,
        o_rx,
        o_ctl_rx,
    ))
}

pub fn add_peer_address(
    o_rt: &P2pRuntime,
    o_swarm: &mut SwarmType,
    o_peer: PeerId,
    s_addr: &str,
) -> Result<(), String> {
    let o_addr: Multiaddr = s_addr
        .parse()
        .map_err(|_| "add_peer_address: addr parse fehler".to_string())?;

    o_swarm.behaviour_mut().rr.add_address(&o_peer, o_addr);

    {
        // connected peers ist nur status, nicht zwingend sofort verbunden
        // aber du siehst: adresse ist gesetzt

        println!("add_peer_address");
    }

    Ok(())
}

pub fn spawn_swarm_task(
    mut o_swarm: SwarmType,
    mut o_rx: mpsc::Receiver<OutboundCall>,
    mut o_ctl_rx: mpsc::Receiver<SwarmControlMsg>,
    o_server_handler: ServerHandler,
    o_connected_peers: Arc<Mutex<HashSet<PeerId>>>,
) {
    tokio::spawn(async move {
        let mut map_pending: HashMap<RequestIdType, oneshot::Sender<Result<Vec<u8>, String>>> =
            HashMap::new();

        loop {
            tokio::select! {
                            o_ctl_opt = o_ctl_rx.recv() => {
                                if let Some(o_ctl) = o_ctl_opt {
                                    match o_ctl {
                                        SwarmControlMsg::RegisterPeerAddr { s_peer_id, s_addr } => {
                                            let o_peer_id = match parse_peer_id(&s_peer_id) {
                                                Ok(v) => v,
                                                Err(e) => { println!("register: {}", e); continue; }
                                            };
                                            let o_addr = match parse_multiaddr(&s_addr) {
                                                Ok(v) => v,
                                                Err(e) => { println!("register: {}", e); continue; }
                                            };
                                            o_swarm.behaviour_mut().rr.add_address(&o_peer_id, o_addr);
                                            println!("register: peer={} addr gesetzt", s_peer_id);
                                        }
                                        SwarmControlMsg::ConnectPeer { s_peer_id, s_addr } => {
                                            let o_peer_id = match parse_peer_id(&s_peer_id) {
                                                Ok(v) => v,
                                                Err(e) => { println!("connect: {}", e); continue; }
                                            };
                                            let o_addr = match parse_multiaddr(&s_addr) {
                                                Ok(v) => v,
                                                Err(e) => { println!("connect: {}", e); continue; }
                                            };

                                            o_swarm.behaviour_mut().rr.add_address(&o_peer_id, o_addr.clone());

                                            match libp2p::swarm::Swarm::dial(&mut o_swarm, o_addr) {
                                                Ok(_) => println!("connect: dial gestartet peer={}", s_peer_id),
                                                Err(e) => println!("connect: dial fehlgeschlagen: {}", e),
                                            }
                                        }
                                    }
                                } else {
                                    break;
                                }
                            }

                            o_call_opt = o_rx.recv() => {
                                if let Some(o_call) = o_call_opt {
                                    let i_id = o_swarm
                                        .behaviour_mut()
                                        .rr
                                        .send_request(&o_call.o_peer, o_call.v_req);

                                    map_pending.insert(i_id, o_call.o_tx);
                                } else {
                                    break;
                                }
                            }

                            o_evt = o_swarm.select_next_some() => {
                                match o_evt {

                                    SwarmEvent::NewListenAddr { address, .. } => {
                                        println!("listen addr: {}", address);
                                    }

                                    SwarmEvent::ListenerError { error, .. } => {
                                        println!("listener error: {}", error);
                                    }

                                    SwarmEvent::ListenerClosed { .. } => {
                                        println!("listener closed");
                                    }

                                    SwarmEvent::IncomingConnection { send_back_addr, .. } => {
                                        println!("incoming connection from: {}", send_back_addr);
                                    }

                                     SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                                        {
                                            let mut set_peers = o_connected_peers.lock().await;
                                            set_peers.insert(peer_id);
                                        }
                                        print_connected_peers(&o_connected_peers).await;
                                    }
                                    SwarmEvent::ConnectionClosed { peer_id, .. } => {
                                        {
                                            let mut set_peers = o_connected_peers.lock().await;
                                            set_peers.remove(&peer_id);
                                        }
                                        print_connected_peers(&o_connected_peers).await;
                                    }

                                    SwarmEvent::Behaviour(P2pBehaviourEvent::Mdns(mdns::Event::Discovered(v_list))) => {
                                        for (o_peer, o_addr) in v_list {
                                            o_swarm.behaviour_mut().rr.add_address(&o_peer, o_addr);
                                        }
                                    }

                                    SwarmEvent::Behaviour(P2pBehaviourEvent::Mdns(mdns::Event::Expired(v_list))) => {
                                        for (o_peer, o_addr) in v_list {
                                            o_swarm.behaviour_mut().rr.remove_address(&o_peer, &o_addr);
                                        }
                                    }

                                    SwarmEvent::Behaviour(P2pBehaviourEvent::Rr(request_response::Event::Message { peer, message, .. })) => {
                                        match message {

                                            request_response::Message::Response { request_id, response } => {
                                                if let Some(o_tx) = map_pending.remove(&request_id) {
                                                    let _ = o_tx.send(Ok(response));
                                                }
                                            }

                                            request_response::Message::Request { request, channel, .. } => {
                                                let v_resp = match (o_server_handler)(peer, request).await {
                                                    Ok(v_ok) => v_ok,
                                                    Err(s_err) => s_err.as_bytes().to_vec(),
                                                };

                                                let _ = o_swarm
                                                    .behaviour_mut()
                                                    .rr
                                                    .send_response(channel, v_resp);
                                            }
                                        }
                                    }

                                    _ => {}
                                }
                            }
                        }
        }
    });
}

pub async fn send_request_bytes(
    o_rt: &P2pRuntime,
    o_peer: PeerId,
    v_req: Vec<u8>,
) -> Result<Vec<u8>, String> {
    let (o_tx, o_rx) = oneshot::channel::<Result<Vec<u8>, String>>();

    let o_call = OutboundCall {
        o_peer,
        v_req,
        o_tx,
    };

    o_rt.o_tx
        .send(o_call)
        .await
        .map_err(|_| "p2p send queue closed".to_string())?;

    o_rx.await.map_err(|_| "p2p oneshot closed".to_string())?
}

pub async fn get_connected_peers(o_rt: &P2pRuntime) -> Vec<PeerId> {
    let set_peers = o_rt.o_connected_peers.lock().await;
    set_peers.iter().cloned().collect()
}

async fn print_connected_peers(o_connected_peers: &Arc<Mutex<HashSet<PeerId>>>) {
    let set_peers = o_connected_peers.lock().await;
    println!("------------------------------------------------------------");
    println!("connected peers count: {}", set_peers.len());
    for o_peer in set_peers.iter() {
        println!("peer: {}", o_peer);
    }
    println!("------------------------------------------------------------");
}

pub fn connect_peer_manual(
    o_swarm: &mut SwarmType,
    o_peer_id: PeerId,
    s_addr: &str,
) -> Result<(), String> {
    let o_addr: Multiaddr = s_addr
        .trim()
        .parse()
        .map_err(|_| "connect: addr parse fehler".to_string())?;

    // adresse fuer request response setzen
    o_swarm
        .behaviour_mut()
        .rr
        .add_address(&o_peer_id, o_addr.clone());

    // dial starten
    Swarm::dial(o_swarm, o_addr).map_err(|e| format!("connect: dial fehlgeschlagen: {}", e))?;

    Ok(())
}
