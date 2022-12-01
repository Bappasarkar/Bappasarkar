import splitfolders

input_folder='/home/kawsar/aA-WORKING/archive/RegenaratedX/Brinjal/'
output_folder='/home/kawsar/aA-WORKING/archive/TrainDataFolder/Brinjal/'
splitfolders.ratio(input_folder,output_folder,seed=42,ratio=(.7,.2,.1))