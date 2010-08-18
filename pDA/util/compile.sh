#!/bin/bash
echo "=============================================="
echo "Compiling files..."
javac -cp $DXC_HOME/Src/APIs/java/src da/*.java
echo "done."
echo "=============================================="

echo "\n=============================================="
echo "Removing old Package Archives & Configuration..."
rm bin/*
echo "done."
echo "=============================================="

echo "\n=============================================="
echo "Packaging new files into Archive..."
cd da/
jar cvf DiagnosticAlgorithm.jar *.class
cd ../
echo "done."
echo "=============================================="

echo "\n=============================================="
echo "Moving Package Archive & Configuration..."
cp da/*.jar bin/
cp da/*.xml bin/
echo "done."
echo "=============================================="

echo "\n=============================================="
echo "Cleaning up..."
rm da/*.class
rm da/*.jar
echo "done."
echo "=============================================="

echo "\n=============================================="
echo "Job Complete"
echo "=============================================="
