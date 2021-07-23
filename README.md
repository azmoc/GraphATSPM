# GraphATSPM

This program turns a graph of traffic volume into data values. The UDOT ATSPM website provides graphs of traffic metrics but does not provide the data used to create the graphs. One such useful metric is traffic volume for an intersection. The UDOT ATSPM website provides a jpeg of the graph. This code reads it and saves a CSV file in an existing folder called "Values/".
The graph must at least and only show northbound and southbound traffic volumes.
Can take arguments of jpegs' relative file paths
Can take argument "show" to also display comparison graph and table of extracted values.
Can take argument "save" to also save jpeg of comparison graph and table.
Some parameters may need adjustment depending on the graph.