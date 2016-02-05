TODOs
===

* Figure out a better way of re-weighting gradients filtered through a vis path.
  Right now expanding the hierarchy from fc8 to conv1 leads to oversaturated
  images deeper in the tree. There should be a way to adjust gradient magnitudes
  to prevent over-saturation.

* (Bug) Note that caffe must be build without CuDNN support or the ReLU needs to
  be changed to use my implementation. A better way to implement this would
  be in the fashion of yosinski's deconv implementation.

* Allow drilling up in the hierarchy as well as drilling down.

* Implement efficient batch computation. Have VisTree run an event loop.
  Whenever a feature vis is requested it enqueues its path
  and waits for that vis to be computed. Each job in the event loop
  dequeues a whole batch from the queue and runs it. Once a specific path's
  job gets to execute it's likely to have already been handled by some other
  job, so it doesn't have to do any work. Tricky, but fun... use asyncio

* Make both net and image context apparent in URL. There should also be a way to
  select a network from a list of networks that can be visualized.

* Clean up FeatBuilder code.

* The front end needs a lot of work...
