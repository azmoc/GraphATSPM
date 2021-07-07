# GraphATSPM

This program turns a type of graph into data values. The UDOT ATSPM website provides graphs of traffic metrics but does not provide the data used to create the graphs. A useful metric is traffic volume for an intersection. The UDOT ATSPM website provides a jpeg of the graph, which this code reads and saves a CSV file in an existing folder called "Values/".
Can take argument of the jpeg's relative file path
Can take argument "TRUE" to display comparison graph of extracted values.

Possible improvements:
The algorithm is only mostly accurate because it averages a cluster of pixels to determine a graph value. A new algorithm that can read the slope of each graph line would really get the job done.