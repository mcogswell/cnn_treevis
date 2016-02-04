release TODOs
===

* Revise the REST API so it posts json instead of sending GET keys.
  This allows the API to reflect the full semantics (a reconstruction
  is in the context of all its ancestors in the tree... it's not just specific
  to one net/image/neuron tuple... rather it's net/image/root_neuron/child1/child2/..../this_neuron...
  that information should be specified as json. This may allow me to get rid of the giant
  json_tree object that gets passed around so the client only has to deal with local information.







---


This is a pre-alpha quality release of visualization software for Convolutional Neural Networks (CNNs).
Its goal is to help the user understand the hierarchy of parts encoded by a CNN through
visualization of the parts and their relation to existing low and high level intutions.

Application Overview
===

Typical flow

1. open gallery

2. pick an image

3. pick an interesting neuron

4. click on leaf nodes to see how they relate to 


NOTE: This interface could be a lot better. Let me know if you have ideas or
time to work on it.


Frontend Components
===

The main vis is in templates/vis.html.

The other pages... TODO


Backend Components
===

Visualization components are served from a [Flask](http://flask.pocoo.org/) app.
This interacts with a dynamic component which generates example-specific
visualization components and a static/cached component which serves network-specific
components.

Flask App
---
This serves each page in the interface as well as the visualization components within a page.
Each page/component is retrieved through a separate HTTP request.

TODO: detail web api here or lower in the page


Dynamic Vis Generator
---
All of the Zeiler/Fergus or Guided Backprop based images are generated at
runtime. Each vis page is associated with a `VisTree` instance (`lib/recon/reconstruct.py`) 
which stores and computes vis components for one image.


Cached Vis Generator
---
Some parts of the application are generated before runtime and cached
as static content. Currently, this only includes neuron-wise visualizations,
which must be generated offline with a command line utility.







TODOs / Bugs
===

* Note that caffe must be build without CuDNN support or the ReLU needs to
  be changed to use my implementation. A better way to implement this would
  be in the fashion of yosinski's deconv implementation.

* [Don't preserve context when it should change]
  If I (1) vis an activation and investigate its hierarchy to find some child activation
  then (2) vis the child activation independently, then the new child activation vis
  will be the same one (with the same context) as the child vis as a leaf in the hierarchy
  of the original activation vis (i.e. not a single pixel, but all pixels that came from
  backpropping the original activation vis to the child neuron). Restarting the
  server fixes this, but I shouldn't have to do that. The problem is that it caches
  activations based on blob name and feature id, but different contexts lead to different
  visualizations for the same blob name and feature id.


TODOs / Features
===

* Allow drilling up in the hierarchy as well as drilling down.


TODOs / Release
===

* Read through and delete notes.md

* Make both net and image context apparent in URL
