Pipeline the computation and memory access processes to hide latency.

In order to overlap the latency of memory access and fully utilize the processor's computation, an additional set of memory is requested to load the data needed for the next round of computation in advance, avoiding the blocking caused by memory access.

The specific describe of the double buffer is as follows:

{describe}

The example code is as follows:

{code}
