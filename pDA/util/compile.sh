#!/bin/bash
echo "=============================================="
echo "Compiling files..."
javac -verbose -cp $DXC_HOME/Src/APIs/java/src da/gov/dod/army/rdecom/tardec/tcctda/*.java
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
jar cvf DiagnosticAlgorithm.jar gov/dod/army/rdecom/tardec/tcctda/*.class
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
rm da/gov/dod/army/rdecom/tardec/tcctda/*.class
rm da/*.jar
echo "done."
echo "=============================================="

echo "\n=============================================="
echo "Job Complete"
echo "=============================================="
