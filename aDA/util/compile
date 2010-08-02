#!/bin/bash
echo "\nCompiling files..."
javac -cp ../da DiagnosticAlgorithm.java ErrorFinder.java Sensor.java
#javac -cp $DXC_HOME/Src/APIs/java/src MyDA2.java Sensor.java ErrorFinder.java
echo "\ndone."

echo -e "\nPacking files..."
jar cvf ../MyDA2.jar *.class
echo "\ndone."

echo "\nCleaning up..."
rm ../da/*.class
echo "\ndone."

echo "/n/nJob Complete"
