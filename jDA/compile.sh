#!/bin/bash
echo "Compiling files..."
javac -cp $DXC_HOME/Src/APIs/java/src MyDA2.java Sensor.java ErrorFinder.java ComponentError.java
echo "done."

echo -e "\nPacking files..."
jar cvf MyDA2.jar *.class
