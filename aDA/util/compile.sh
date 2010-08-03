#!/bin/bash
echo "=============================================="
echo "Compiling files..."
javac -cp $DXC_HOME/Src/APIs/java/src ../da/*.java
echo "done."
echo "=============================================="

echo "\n=============================================="
echo "Removing old Package Archives..."
rm ../bin/*.jar
echo "done."
echo "=============================================="

echo "\n=============================================="
echo "Packaging new files into Archive..."
jar cvf ../bin/DiagnosticAlgorithm.jar ../da/*.class
echo "done."
echo "=============================================="

echo "\n=============================================="
echo "Cleaning up..."
rm ../da/*.class
echo "done."
echo "=============================================="

echo "\n=============================================="
echo "Job Complete"
echo "=============================================="
