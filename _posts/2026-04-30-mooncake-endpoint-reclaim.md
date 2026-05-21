---
layout: post
title: 'Mooncake''s RDMA QP Leak: A Drain-Coupling Pathology'
---

<div class="tikz-figure">
<script type="text/tikz">
\usetikzlibrary{arrows.meta,positioning,fit,calc}
\begin{tikzpicture}[
  >={Stealth[length=2mm]},
  font=\small,
  box/.style={draw, align=center, inner sep=4pt, rounded corners=1pt},
  smbox/.style={box, font=\footnotesize},
  edgelbl/.style={font=\scriptsize, fill=white, inner sep=1.5pt},
  edgelblc/.style={edgelbl, align=center},
]
\node[smbox, align=left] (upstream) at (0, 8.2) {
  SGLang PD-disagg: \texttt{batchTransferSync()} fails $\Rightarrow$ Python marks session failed; \\
  \texttt{handle\_map\_} stays cached, \texttt{closeSegment} is no-op $\Rightarrow$ peer's insertions stop
};
\node[smbox] (del) at (-5.5, 5) {\texttt{deleteEndpoint(peer)} \\ \emph{error path}};
\node[smbox] (miss) at (0, 5) {\texttt{RdmaContext::endpoint(peer)} \\ \emph{cache-miss path}};
\node[smbox, line width=0.8pt] (mon) at (5.5, 5) {\texttt{WorkerPool::monitorWorker} \\ \emph{1\,Hz heartbeat, NUMA-pinned} \\ {\scriptsize \texttt{worker\_pool.cpp:447--451}}};
\node[box] (map) at (-3, 2) {
  \texttt{endpoint\_map\_} \\
  {\footnotesize \texttt{unordered\_map<string,}} \\
  {\footnotesize \texttt{shared\_ptr<RdmaEndPoint>>}}
};
\node[box] (wait) at (3, 2) {
  \texttt{waiting\_list\_} \\
  {\footnotesize \texttt{unordered\_set<}} \\
  {\footnotesize \texttt{shared\_ptr<RdmaEndPoint>>}} \\
  {\footnotesize + atomic \texttt{waiting\_list\_len\_}}
};
\node[box, minimum width=120mm] (nic) at (0, -2) {
  NIC hardware QP pool ($\sim$64K slots) \\
  {\footnotesize finite, NIC-scope; exhaustion blast radius is the whole NIC, not the failing peer}
};
\node[box, dashed, align=left, font=\footnotesize] (fail) at (9.5, 2) {
  Pre-fix failure mode: \\
  \quad inserts stall (peer dead) \\
  \quad evict/delete keep firing \\
  \quad reclaim never invoked \\
  \quad \texttt{waiting\_list\_} grows \\
  \quad 1118 evicts $\Rightarrow$ \\
  \quad\quad {>}20K QPs per NIC \\
  \quad \texttt{ibv\_create\_qp} returns \\
  \quad\quad \texttt{ENOMEM} NIC-wide
};
\node[draw, dotted, fit=(map)(wait), inner sep=12pt] (store) {};
\node[font=\footnotesize, anchor=south west] at ([xshift=2pt]store.north west) {EndpointStore (FIFO or SIEVE), \texttt{RWSpinlock endpoint\_map\_lock\_}};
\node[draw, dashed, fit=(del)(miss)(mon)(store), inner sep=12pt] (ctx) {};
\node[font=\footnotesize, anchor=south west] at ([xshift=2pt]ctx.north west) {RdmaContext (one per NIC)};
\draw[->, dotted] (upstream.south) -- (miss.north);
\draw[->] (del.south) -- (store.north west) node[edgelbl, midway, sloped] {moves entry to \texttt{waiting\_list\_}};
\draw[->] (miss) -- (map) node[edgelbl, midway, sloped] {1.~\texttt{insertEndpoint}};
\draw[->, dashed] (miss) -- (wait) node[edgelblc, midway, sloped] {2.~\texttt{reclaimEndpoint} \\ \emph{only pre-fix caller}};
\draw[->] (map) -- (wait) node[edgelbl, midway] {evict (FIFO/SIEVE)};
\draw[->, line width=0.8pt] (mon) to[bend right=10] node[edgelblc, midway, sloped] {\textbf{fix}: \texttt{reclaimEndpoints} \\ \emph{every 1\,s}} (wait);
\draw[->] (map.south) -- (map.south |- nic.north) node[edgelbl, midway] {\texttt{ibv\_create\_qp} $\times$ \texttt{num\_qp\_per\_ep}};
\draw[->] (wait.south) -- (wait.south |- nic.north) node[edgelbl, midway] {dtor $\Rightarrow$ \texttt{ibv\_destroy\_qp}};
\draw[->, dotted] (wait.east) to[bend left=15] (fail.west);
\end{tikzpicture}
</script>
</div>

Mooncake is the production serving platform for Moonshot AI's[^moonshot] Kimi.[^kimi] Its Transfer Engine handles Remote Direct Memory Access (RDMA) data movement between prefill and decode clusters. One `RdmaContext` exists per NIC. Each `RdmaContext` owns an `EndpointStore`: a software cache of `RdmaEndPoint` objects keyed on peer Network Interface Controller (NIC) path, bounded in size by `max_endpoints`. Each `RdmaEndPoint` allocates `num_qp_per_ep` Queue Pairs (QPs) at construction with `ibv_create_qp`, and releases them with `ibv_destroy_qp` from its destructor.[^rdma]

QPs are a finite hardware resource. Each NIC has a fixed pool of QP slots (~64K on modern Mellanox/NVIDIA hardware). Software-side endpoint count and hardware-side QP count are coupled by the relation `qps_allocated = sum(num_qp_per_ep over live RdmaEndPoint instances)`. "Live" here means the C++ object has not been destructed. Equivalently, some `shared_ptr<RdmaEndPoint>` still has a reference count > 0.

Under peer failure load, `RdmaEndPoint` instances accumulate without being destructed. `ibv_destroy_qp` never runs for them. The hardware QP count grows monotonically until the NIC's pool is exhausted and `ibv_create_qp` returns `ENOMEM` for every subsequent caller, not only the failing peer.[^issue][^pr] In our case, a reporter running SGLang PD-disaggregated prefill[^pd] hits this with >20K QPs allocated per NIC, asymmetrically distributed (22K on `mlx5_00–03`, 6K on `mlx5_04–07`). This is consistent with eviction concentration on the NICs routing to a specific failing peer.

We can fix this with a simple addition to an existing thread (`WorkerPool::monitorWorker`, a per-`RdmaContext` worker that already runs a NIC-liveness tick), but the aforementioned bug's mechanism is the important thing to understand: the only production caller of the reclaim path was sequentially adjacent to the insertion path. When insertion could no longer be invoked, reclaim also could no longer be invoked. Endpoints kept moving into a holding set for endpoints awaiting safe destruction (`waiting_list_`) but nothing was able to drain them.

## Endpoint store and `RdmaEndPoint` lifecycle

`EndpointStore` is an abstract base class. There are two implementations: `FIFOEndpointStore` and `SIEVEEndpointStore`.[^sieve] Both maintain two members, guarded by a single `RWSpinlock endpoint_map_lock_`:

- `endpoint_map_`: `std::unordered_map<std::string, std::shared_ptr<RdmaEndPoint>>`, keyed on the peer NIC path. This is the active cache.
- `waiting_list_`: `std::set<std::shared_ptr<RdmaEndPoint>>`. These are the endpoints that have been evicted or explicitly deleted from `endpoint_map_` but cannot yet be destructed because outstanding slice work may still hold raw pointers into them.

Note that the name is a misnomer. It should really be called `store_lock_` or simply `lock_`. This two-stage structure exists because endpoint destruction cannot be synchronous with eviction. `RdmaEndPoint` exposes raw pointers, for example, `slice->rdma.qp_depth` is a `std::atomic<int>*` aliased into `RdmaEndPoint::wr_depth_list_`. Thus, outstanding slices may dereference these for the duration of their work. We can derive the following lifecycle of an `RdmaEndPoint` instance:

1. **Insert**: `EndpointStore::insertEndpoint(peer_nic_path, ctx)` is called from `RdmaContext::endpoint(peer_nic_path)` on cache miss. The store constructs an `RdmaEndPoint`, whose constructor invokes `ibv_create_qp` a total of `num_qp_per_ep` times. The resulting `shared_ptr` is inserted into `endpoint_map_`.
2. **Evict or delete**: This stage arises either from FIFO/SIEVE pressure when `endpoint_map_.size() == max_endpoints` or `deleteEndpoint(peer_nic_path)` explicitly being invoked in error paths. The `shared_ptr` is moved out of `endpoint_map_` into `waiting_list_`. The endpoint is marked inactive with `active_ = false` so that subsequent operations do not select it.
3. **Reclaim**[^quiescent]: `reclaimEndpoint()` walks `waiting_list_` and calls `endpoint->hasOutstandingSlice()` on each entry, erasing each entry that returns false. The local `shared_ptr` drops out of scope when the entry is erased. The destructor is called, and invokes `ibv_destroy_qp` a total of `num_qp_per_ep` times.

Note that until reclaim runs, the QPs remain allocated on the NIC.

## Error analysis

`reclaimEndpoint()` had exactly one production caller before the fix: `RdmaContext::endpoint(peer_nic_path)` in `rdma_context.cpp:355-356`:

```cpp
endpoint = endpoint_store_->insertEndpoint(peer_nic_path, this);
endpoint_store_->reclaimEndpoint();
```

These are two sequential calls under no shared lock. `insertEndpoint` has an internal `WriteGuard` that releases before line 356 begins. Together they insert a new endpoint on cache miss, and then drain `waiting_list_`. There is an implicit invariant here: the rate at which `RdmaContext::endpoint()` produces cache misses is at least the rate at which `evictEndpoint()` and `deleteEndpoint()` move entries to `waiting_list_`. This holds perfectly well under healthy load. However, there are a few cases in which this can fail:

- When a peer process dies or its retry budget exhausts, the transport stops issuing connection attempts to that peer's NIC paths.
- Error handling on failed connections or peer disconnect notifications call `deleteEndpoint(peer_nic_path)` explicitly.
- FIFO and SIEVE pressure from healthy peers can continue to evict cached entries with `evictEndpoint()`. Each of these adds to `waiting_list_`.

Since none of these paths call `reclaimEndpoint()`, `waiting_list_` grows monotonically because no remover exists. The original issue encountered this state. Once `ibv_create_qp` returns `ENOMEM`, every subsequent endpoint construction fails, exhausting even peers that are healthy, because the exhaustion is a property of the NIC, and not any peer. The blast radius becomes that of the NIC itself.

## Diagnosis from data

Three signals localized the leak:

1. `rdma resource show` reported >20K QPs allocated per NIC. This is several orders of magnitude above healthy endpoint count. We can infer QPs were being created but not destroyed.
2. The leak was distributed asymmetrically. A uniform leak would distribute evenly, for example a refcount bug affecting every endpoint. The asymmetry implies that the leak rate scales with eviction/deletion load, and that load is concentrated on the NICs routing to the failing peer.
3. There were 1118 `endpoint-evicted` log lines preceding the `ENOMEM`. From this we can infer that construction was working well enough, but something was wrong with destruction.

The conjunction of these 3 signs helped to localize the missing step to the reclaim mechanism. A close reading of `reclaimEndpoint()` confirmed that there was only one production caller, sequentially adjacent to insertion as shown above, and thus reclaim liveness was dependent on insertion liveness.

## Solution

Decoupling insertion from reclamation removes the error. `WorkerPool::monitorWorker` already runs once per `RdmaContext`, NUMA-pinned[^numa] by `bindToSocket(numa_socket_id_)`, on a once-per-second cadence to update `active_=true` as a NIC liveness signal:

```cpp
void WorkerPool::monitorWorker() {
  bindToSocket(numa_socket_id_);
  auto last_reset_ts = getCurrentTimeInNano();
  while (workers_running_) {
    auto current_ts = getCurrentTimeInNano();
    if (current_ts - last_reset_ts > 1000000000ll) {
      context_.set_active(true);
      context_.reclaimEndpoints();
      last_reset_ts = current_ts;
    }
    // `epoll_wait`, async event handling, and so forth.
  }
}
```

The added line `context_.reclaimEndpoints()` is a thin forwarding method on `RdmaContext` that calls `endpoint_store_->reclaimEndpoint()`. The body of `reclaimEndpoint()` remains unchanged. The error is removed by changing where and when it is called.

## Conclusion

The pattern of cleanup only triggering in a creation-adjacent call site is locally attractive because it requires no additional thread, timer, or scheduling primitive. It is correct under the invariant $r_{\text{create}} \geq r_{\text{destroy}}$ over every relevant window of time. It fails only when the particular failure modes violate the invariant.

There are two valid alternatives to this pattern:

1. **Independent driver**: a periodic tick whose liveness is independent of the creation path.
2. **Direct destruction-path drive**: an eager cleanup call from the destruction call site itself that doesn't batch on creation.

The first fits when destruction must be deferred. Mooncake has outstanding raw pointer hazards with `wr_depth_list_`. The second fits when destruction can be synchronous. An earlier failed solution attempted this with an eager `disconnect()` from both `evictEndpoint()` and `deleteEndpoint()`, but it crashed under multi-process `rxe` with `malloc(): unaligned tcache chunk detected` within a few seconds. `slice->rdma.qp_depth` is a raw pointer into `wr_depth_list_` and eager destruction races against slice work in progress. A shared ownership refactor of `wr_depth_list_` would correctly handle this but was outside the scope of this work.

[^numa]: Modern multi-socket servers have NICs attached via PCIe (Peripheral Component Interconnect Express) to a specific CPU socket. Memory accesses from threads running on other sockets cross the inter-socket interconnect. This pays a latency cost and contends for limited cross-socket bandwidth. NUMA (Non-Uniform Memory Access) solves this problem by pinning the per-NIC worker to its NUMA-local socket with `bindToSocket(numa_socket_id_)`. This keeps both the thread's stack and its accesses to `RdmaContext` and `EndpointStore` state in the local memory bank. For background, see Christoph Lameter, ["NUMA (Non-Uniform Memory Access): An Overview"](https://queue.acm.org/detail.cfm?id=2513149).

[^quiescent]: An entry on `waiting_list_` whose `hasOutstandingSlice()` returns false is in a *quiescent* state. No in-flight operation can still observe it, so destruction is safe. The general pattern (logically remove now, free once no observer remains) is *deferred* or *safe memory reclamation*: well-known instances are RCU, epoch-based reclamation, and hazard pointers. For reference see Dijkstra et al., ["On-the-Fly Garbage Collection: An Exercise in Cooperation"](https://www.cs.utexas.edu/~EWD/transcriptions/EWD06xx/EWD630.html) (CACM 1978), where mutator threads are defined as *quiescent* when not mid-pointer-update.

[^issue]: Issue [kvcache-ai/Mooncake#1845](https://github.com/kvcache-ai/Mooncake/issues/1845).

[^pr]: Fix in [PR #1952](https://github.com/kvcache-ai/Mooncake/pull/1952), merged 2026-04-27 at `489c0207`.

[^sieve]: FIFO goes without explanation. [SIEVE](https://www.usenix.org/conference/nsdi24/presentation/zhang-yazhuo) carries a `visited` bit on access. A hand pointer walks the queue, clearing the bit on visited entries, giving them a second chance, and evicting the first unvisited one. This result approximates LRU's hit rate with the same cost as FIFO per operation. For this bug it is irrelevant, however. The leak is in the reclaim path that is invoked after the particular eviction algorithm has chosen a victim.

[^moonshot]: [Moonshot AI](https://www.moonshot.ai/).

[^kimi]: [Kimi](https://www.kimi.com/).

[^rdma]: For background on RDMA verbs and the `ibv_*` API surface, see Dotan Barak's [RDMAmojo](https://www.rdmamojo.com/). For design tradeoffs in production RDMA systems, see Kalia et al., ["Design Guidelines for High Performance RDMA Systems"](https://www.usenix.org/conference/atc16/technical-sessions/presentation/kalia) (USENIX ATC 2016).

[^pd]: Prefill-Decode disaggregation runs the two phases of LLM inference on separate GPU pools. Prefill (processing the prompt to build the KV cache) is compute-bound; decode (generating tokens one at a time) is memory-bandwidth-bound. Co-locating both wastes one resource or the other, so disaggregated topologies route each request prefill -> decode and ship the KV cache between them, which is what Mooncake's Transfer Engine handles. A prefill node maintains one `RdmaEndPoint` per decode peer it serves. See Zhong et al., ["DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) (OSDI 2024) for the canonical treatment, and Qin et al., ["Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving"](https://arxiv.org/abs/2407.00079) for Mooncake's specific design.
