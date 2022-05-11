# GraphToCSV.py

This program turns a graph of traffic approach volume into data values. The UDOT ATSPM website provides graphs of traffic metrics but does not provide the raw data. In particular, the UDOT ATSPM website provides a jpeg of the graph of approach volume for an intersection. This code reads a jpeg and writes a CSV file in an existing folder called "Volumes/".

Example Command Line (powershell terminal):
- python graphToCSV.py show Graphs2020-01/VolumesGraph2020-01-01.jpg save overlay

Command Line Arguments
- String arguments of one or more jpegs' relative file paths.
- Optional string argument "show" to also show comparison graph and table of extracted values.
- Optional string argument "save" to also save jpeg of comparison graph and table.
- Optional string argument "overlay" to also overlay the comparison graph on the original.
- Optional string argument "5" to assume 5 minute bins instead of 15 minute bins.

Download jpegs of traffic approach volume from the following URL:
https://udottraffic.utah.gov/atspm/
- The graph must show two directions' (e.g. northbound and southbound) traffic volumes.
- Do not include directional split.
- Recommended use for 15 minute bins though it can take 5 minute bins.

The destination folders must exist prior to running - extracted data and plots go to "Volumes/" and "Plots/" respectively.

Code quirks:
- Some internal parameters may need adjustment depending on the graph.
- If no filename argument is given, 31 jpegs are read from "Graphs2020Jan/Graphs2020Jan/VolumesGraph2020-01-xx.jpg" with xx filled by 01 to 31.
- If "show" is in arguments, it saves the volume data to "Volumes/VolumesXxxxxxxxxX.csv" where the Xx...xX is filled by the last 10 digits of the jpeg name (not including .jpg). It is best to make it the date in format YYYY-MM-DD.
- Tested on data of traffic signal 6415.
