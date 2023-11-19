# Location Detection Algorithm for ESID projects

## Installation:
- Create Conda environment ```create -n location_detection python=3.9```
- Activate Conda environment ```conda activate location_detection```
- CD into the requirements.txt file path 
- Install the requirements.txt ```pip install -r requirements.txt```
- To test if the model is working, run ```python demo.py```

## Command line usage:
Use ```python main.py``` to run the tool with the following parameters:
- ```-c``` MongoDB collection name, by default it is set to ```all_newcrawl_combined_271022```
- ```-p``` Path to a text file of project IDs to predict their locations. By default, if this value is not passed, the model will select all the projects from MongoDB and predict their locations, and insert their location into  MySQL database.

