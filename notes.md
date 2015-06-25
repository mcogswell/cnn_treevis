* cite [leopard sofa blog](https://rocknrollnerd.github.io/ml/2015/05/27/leopard-sofa.html)
    * point out that the CNN really does look at parts




software requirements
===
input: any image
output: visualization
    * predicted image class
    * shows "why the cnn made that classification" through a hierarchical break down of neurons
    ...
    * should have a static mode (for the blog)
    * (extra) should have a dynamic mode (for exploring)
    * (extra) should be easy for others to host on their gpu-enabled machines
    * (extra) visualize imagenet images and show true image classes... this could help find flaws

how does the vis work
    parallel coordinates... each coordinate is a layer of the network
    polylines through the coordinates come from a tree whose root is at the visualized neuron
    to construct the tree, start with a nueron called the current neuron (here, the unit for the most probable class)
    (a) form a list L of top-k maximally activated neurons in the previous layer
    form an edge between each of the neurons in L and the current neuron
    at each node of L, if a feature map, highlight the corresponding image patch
        * api call
            input:
              image, current net, blob name, feature index
            output:
              (find the maximally activated pixel in that feature map and mask a region around it)
              image slightly greyed out except mask region
    at each node of L, display that node's canonical image (to give a sense of what activates that neuron)
        * api call
            input:
              current net, blob name, feature index, k
            output:
              zeiler reconstructions for the top-k images
              corresponding patches with a small window of context
        (extra) also show the top classes which activate that neuron
    also display the zeiler vis of that neuron FOR THE CURRENT IMAGE
        * api call
            input:
              image, current net, blob name, feature index
            output:
              zeiler reconstruction
              bounding box corresponding to patch in original image
    for each neuron, recurse according to (a)

(extra) interactions
    click on an activation
    vis highlights a multi-branching path through parallel coordinates
        coordinates == network layers
    one path traces back through neurons which were highly activated

(extra) interaction
    when a canonical image is clicked on... go to the vis with that image as the example
