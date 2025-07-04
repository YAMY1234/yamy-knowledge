---
id: distributed-parallelism
sidebar_position: 4
title: 多机并行机制
---
## AllReduce通信协议

## **定义:**

AllReduce是一种分布式通信原语，执行"**全体求和并广播**"操作:将参与进程(or GPU)上各自持有的数值张量求和，并将结果分发回所有进程，使得每个进程都得到和。在深度学习中，多机数据并行训练使用AllReduce来汇总不同GPU计算的梯度平均值，从而更新模型参数同步一致。

## **原理:**

AllReduce可以用多种算法实现，其中最经典高效的是 **环形AllReduce**(Ring-AllReduce)。假设有N个节点，每个节点有一个数据块。环算法流程是:每个节点将自己的数据按块拆分成N段，环形发送并累加:在前N-1轮中，每轮每个节点将收到来自前一节点的一段累加到本地，然后转发给下一节点。这样循环一圈后，每段求和结果驻留在某节点。接着后N-1轮进行 **广播**:每轮每节点将持有的某段结果发送给下一节点，最终每个节点收齐所有段的总和。整个过程通信量为2*(N-1)次发送/接收，与参数服务器集中发送的全对全通信相比效率更高、带宽利用更优化。

## **用途:**

AllReduce协议广泛用于同步分布式训练的梯度平均。此外在参数服务器(PS)架构以外，Horovod等框架通过AllReduce避免了参数服务器瓶颈。NCCL库内部也实现了基于GPU直连拓扑的高效AllReduce，使得在单机多卡和多机GPU集群中梯度同步达到接近硬件带宽的吞吐。除梯度外，一些需要全局汇总的指标计算(如准确率平均)也可用AllReduce实现。

## **注意事项:**

AllReduce假定参与各方都能存储整个结果，适用于数据并行梯度这样所有节点需要同步相同结果的场景。如果数据量非常大导致内存不敷，也可用AllReduce的分片变体(but generally gradients are fine)。要注意AllReduce需要参与者同步等待，如果有节点特别慢，会拖累整体(synchronization mechanism's inherent problem)。NCCL的环AllReduce已经非常高效，但在不同网络拓扑下可能选择树形AllReduce等算法。某些通信优化技巧如压缩梯度或局部聚合可以减少AllReduce压力，但会带来近似误差。总的来说，AllReduce协议是理解各种分布式同步算法的基础，它的正确实现和调优直接影响大规模训练的可扩展性。

## Ring-AllReduce、NCCL 与 BSP/异步机制

## **Ring/NCCL机制:**

正如上节所述，环形AllReduce是目前分布式训练中应用最广的梯度同步算法，由于其通信均摊到各节点，没有集中瓶颈，扩展性好。NVIDIA的NCCL库对Ring-AllReduce做了高度优化，能根据GPU拓扑(PCIe、NVLink、InfiniBand等)选择最佳通信路径并并行化传输，从而实现接近硬件带宽的性能。NCCL还支持半双工、全双工、树形等多种AllReduce实现，在不同GPU拓扑下优化延迟。通过Horovod、PyTorch DDP等调用NCCL，用户无需关心底层细节即可完成高效同步。总而言之，Ring-AllReduce+NCCL已成为同步SGD通信的事实标准方案。

## **BSP(Bulk Synchronous Parallel):**

批量同步并行，是分布式训练的一种执行模式。其要点是在每次迭代中，各并行节点独立完成计算阶段(如前向后向)，然后在同步点汇合，交换梯度并同步更新，再进入下一迭代。也就是说所有节点严格"对齐"同步。数据并行+AllReduce策略天然属于BSP范式，每次AllReduce就是同步屏障。BSP的优点是训练过程与单机等价，收敛稳定易分析；缺点是易受慢节点拖累，"最长木板效应"显著。

## **异步并行:**

与BSP对应的是异步并行(Asynchronous Parallel)，常通过参数服务器架构实现。每个计算节点不等其他节点完成就将梯度发送到参数服务器，参数服务器立即应用更新并返回新的参数给下一个计算请求。这样各节点更新模型时可能使用的参数不是完全同步的(可能带有滞后)。好处是不会因为个别慢节点阻塞整体，计算资源利用率更高；坏处是"**陈旧梯度**"问题会影响模型收敛，甚至需要特别算法(如对学习率做校正)才能保证稳定性。Hogwild算法是极端的异步SGD，实现多个线程无锁更新参数，效率高但在深度学习大规模场景不直接适用。

## **应用场景对比:**

BSP适合中小规模集群且需要精确同步的训练，比如CV、NLP模型典型的十几到几百GPU训练，一般采用BSP的AllReduce。异步并行在超大规模或者跨设备不均匀网络中曾被采用，如早期Google Parameter Server用于上千CPU训练大模型。但异步收敛难以调校，目前主流深度学习大多回归BSP或接近同步的模式(如Facebook Chainer优化在一定迭代内做局部同步，属于近似BSP)。

## **注意事项:**

如果遇到跨机延迟极高或者节点易掉线的环境，异步方案可能在吞吐上更有利，但需投入更多在算法(学习率、稀疏更新等)调优上。对于绝大多数大模型训练，优先考虑优化同步训练(如Gradient Compression、混部调度)而非轻易改为异步，因为异步的收敛不确定性风险较大。综上，Ring-AllReduce/NCCL+BSP代表了当前性能和收敛稳定性的折中，而异步并行是特殊场景下为追求硬件利用率的方案，需谨慎选择。

## 拓扑感知调度

## **定义:**

在多机多GPU集群中，调度作业或放置进程时考虑底层硬件拓扑结构(如GPU之间/节点之间的通信距离和带宽)，从而将需要大量互相通信的任务尽量安排在拓扑距离近的硬件上，减少通信开销。这称为拓扑感知(Topology-aware)的调度或放置。

## **用途:**

在分布式训练中，不同GPU/节点间带宽差异很大(同一机器GPU间通过NVLink，带宽高延迟低；跨机器通过InfiniBand/以太网，带宽低延迟高)。拓扑感知调度可以显著提升分布式训练性能:

* **GPU进程绑定:** 如果一个节点有8卡GPU且按NVSwitch互联为两组4卡，训练一个8卡任务时，拓扑感知工具会智能绑定MPI进程到GPU，使得环AllReduce的通信流经NVSwitch内部而非跨CPU PCIe，从而降低GPU间通信时间。NVIDIA的NVTAGS就是用于MPI作业的GPU拓扑自动选择工具，据称可提高高通信密集型HPC作业的性能最多达50%以上。
* **节点选择和打包:** 在有多个节点的集群提交作业时，拓扑感知调度器会倾向把同一个训练任务的GPU放在尽可能少的机器内，或者在同一机架/交换机内。如果任务需要16张GPU，理想情况是打包在2台8卡服务器而非分散4台4卡服务器，因为后者有更多跨机器通信。Volcano等调度器支持Network Topology Aware策略，会优先在同一个"网络拓扑区域"内满足Pod的GPU需求，再不得已才跨区。这减少了跨交换机通信，性能更稳定。

## **典型实现:**

Kubernetes中可以通过Topology Aware Scheduling插件或自定义调度实现。例如阿里云ACK提供GPU拓扑感知调度功能，可在节点打标签启用。提交MPI作业时加参数`--gputopology=true`，调度器会选择GPU布局最佳的节点组合。Slurm等传统调度器也有拓扑权重选项或与HWLOC结合决定进程绑定。此外训练框架内部也做相应优化:NCCL在2.7+版本引入了拓扑检测，会自动选择树AllReduce或环AllReduce并安排通信顺序，以匹配GPU/NIC拓扑达到更优效率。

## **注意事项:**

拓扑感知可能与传统的"均匀分配资源"相冲突。例如为了性能可能需要把多个重任务挤在一台服务器而留另一台空闲，这在非拓扑调度中不常见，但长远看整体运行时间可能更短。需要调度系统在考虑拓扑同时仍兼顾公平。也要确保获取拓扑信息的可靠性，可能需要管理员维护好节点的拓扑标签。NVTAGS等工具运行也有少许开销(如对GPU进行一次拓扑profiling)，但通常是一次性的。另外对于跨机训练，如果网络结构复杂(多层交换)，调度策略也需更智能，例如按机架聚集。这往往需要调度系统有全局网络拓扑视图。总的来说，拓扑感知调度是优化分布式训练"最后一公里"的重要措施，在大规模GPU集群中可以带来可观的性能提升。

## 跨Region训练与同步

## **定义:**

跨Region指跨越地理上相距较远的数据中心进行协同训练。由于不同地域机房间网络延迟和带宽相较单数据中心内差异巨大，跨Region训练被视为极端情况下为利用全球算力而采取的特殊并行方案。这通常需要在同步频率和通信模式上进行调整，以减少远程通信开销。

## **挑战:**

广域网(WAN)延迟动辄数十毫秒，比单机房内高几个数量级。直接用同步SGD跨Region，等待远端梯度会严重拖慢训练。例如1000公里外两个数据中心光传播往返就约10ms，加上设备延迟，可能每步同步多出几十毫秒。对于一个本地1ms能做完AllReduce的任务来说，这是不可接受的损耗。因此需要特殊策略让大部分同步在本地进行，只少量信息跨Region传输。

## 可能方案
* **层次化同步(Hierarchical All-Reduce):** 将全球集群按Region分组，各Region内部频繁进行梯度AllReduce，然后各Region之间较低频率地汇总同步。例如一个想法是每训练10步只在Region内部同步梯度，第10步结束时再把各Region的模型权重做一次平均(这类似Local SGD思想)。这样远程通信频率降低了10倍。Facebook等研究提出的LocalSGD(又称DILOCO等)证明，每隔一定步骤再全局同步仍可维持收敛。DeepMind的DiLoCo算法就是每500步才交换一次"伪梯度"，将通信量降低了500倍，同时收敛效果与完全同步近似。
* **流水并行跨站:** 另一思路是把模型拆分，让不同Region负责不同部分，减少同步需求。例如管道并行时，可以设定一个分区边界在这里同步较少参数。当然，这需要模型结构配合且仍存在跨Region激活通信。
* **异步+联邦式更新:** 较为极端的方法是各Region独立训练一段时间，然后异步地交换参数或梯度调整。这接近联邦学习思想，让每个数据中心在自己本地数据上训练epoch后再融合。虽然严格来讲这已不是同时训练一个模型，而更像并行多模型周期性聚合。
* **硬件优化:** 建设专用的低延迟跨洲光纤网络，加装中继放大器和先进路由，使得跨Region延迟和带宽尽可能逼近理想值。例如OpenAI和微软计划通过超大规模光纤互联多座超算中心实现全球一体训练。这些硬件投入非常巨大，但长远看或许可行。

## **实际案例:**

据报道，Google 已尝试跨数据中心训练如Gemini等大模型，方法是将训练作业整体调度到不同数据中心以平衡负载，而非真正拆分一个作业跨Region运行。Meta的MAST调度可以在区域过载时把一些训练任务转移到另一Region，但仍保持单任务不跨Region。这说明目前工业界也趋向于避免真正跨Region同步，而是通过全局调度来优化资源利用。

## **注意事项:**

跨Region训练首要风险是收敛失败或变慢。如果降低同步频率，需要重新调节学习率、批大小等超参，否则可能收敛变差甚至发散。此外，跨Region流量巨大，成本昂贵且易受网络抖动影响，要有容错机制，如同步超时重试或逐步减少远程节点权重。如果可以，尽量将训练限制在网络延迟在可控范围(低于5毫秒)的区域内(例如同一大洲的多数据中心)。总之，跨Region同步是"不得已而为之"的超大规模方案，实现难度高。除非拥有全球布局算力且模型超出单Region容量，否则更提倡在单Region内完成训练，或采用上述层次同步确保大部分通信在低延迟环境中进行。
