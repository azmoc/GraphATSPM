# GraphToCSV.py

This code turns a graph of traffic approach volume into data values. The Utah Department of Transportation (UDOT) Automated Traffic Signal Performance Measures (ATSPM) website provides graphs of traffic metrics but does not provide the raw data. In particular, it provides a JPEG of the graph of daily approach volumes for an intersection. This code reads a JPEG and writes a CSV file in an existing folder called "Volumes/".

![alt text](./Example/ApproachVolumesGraph2020-01-01.jpg)
![alt text](./Example/extracted2020-01-01.jpg)

Example to run the program from Command Line (powershell terminal):
- >python graphToCSV.py Graphs2020-01/VolumesGraph2020-01-01.jpg
- >python graphToCSV.py Graphs2020-01/VolumesGraph2020-01-01.jpg show save overlay 5 table

Command Line Arguments (order does not matter):
- String arguments of one or more jpegs' relative file paths.
- String argument (optional) "show" to also show comparison graph of extracted values.
- String argument (optional) "save" to also save JPEG of comparison graph.
- String argument (optional) "table" to also include table of values below comparison graph.
- String argument (optional) "5" to use 5 minute bins instead of 15 minute bins.
- String argument (optional) "overlay" to also overlay the comparison graph on the original.

Download jpegs of traffic approach volume from the following URL:
https://udottraffic.utah.gov/atspm/
- The graph must show two directions (e.g. northbound and southbound) of traffic volumes.
- Do not include directional split.
- Recommended use for 15 minute bins though it can take 5 minute bins.

The destination folders must exist prior to running. Extracted data and plots go to "Volumes/" and "Plots/" respectively.

Code quirks:
- It writes the volumes data to "Volumes/VolumesXxxxxxxxxX.csv" where the Xx...xX is filled by the last 10 digits of the jpeg filename (not including .jpg). It is best to make it the date in format YYYY-MM-DD.
- If no filename argument is given, 31 jpegs are read from "Graphs2020-01/ApproachVolumesGraph2020-01-xx.jpg" with xx filled by 01 to 31. Then their data is saved to "Volumes/6415Volumes2020-01.csv".
- Tested on data of traffic signal 6415.
- Some internal parameters may need adjustment depending on the graph.
