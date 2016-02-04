release TODOs
===

* Read through and delete notes.md

* Make both net and image context apparent in URL

* Enable specification of GPU

* Can I clone the repo and generate everything required from just ilsvrc data?
    * make this one command

* create gifs and put in README

* Try fooling around with masking intensity options

* make buttons visible sometimes

* go through TODOs in code



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


TODOs / Features
===

* Allow drilling up in the hierarchy as well as drilling down.

