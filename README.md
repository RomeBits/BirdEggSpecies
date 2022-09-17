# Bird Egg Species Identification Research Project
## Researchers: Benjamin Rome, Luke Williams
## Professor Charles V. Stewart

**Goals**

_Primary:_
- Organize and summarize the data. We are most interested in the number of eggs and number of nests per species.  Average is good, but histograms are needed as well. We are particularly interested in performance on species for which we have a small number of examples.
- Segment the eggs from the background and orient the eggs. This can be done by training a network, but I think simple thresholding and connected components and shape analysis will work
- Segment the ruler and turn it into a measurement on the eggs
- Split images into training validation and test. Eggs from the same picture should stay together and not be split up
- Train several different networks to identify and analyze results. Performance measurements on number of examples

_Secondary:_
- How to use the ruler?
- How to combine the query results from multiple eggs in a nest?
- How well can we do when there are new species of eggs that haven’t been seen during training?
- Quantifying performance based on complexity of egg pattern and number of training examples
