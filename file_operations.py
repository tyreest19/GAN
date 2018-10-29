import os, os.path
import shutil


def getAmountOffFilesInDirectory(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

print('amount of Pokemon images', getAmountOffFilesInDirectory('train/Pokemon'))
print('amount of Digimon images', getAmountOffFilesInDirectory('train/Digimon'))

def moveFilesToTestDirectory(directory):
    """This function will move 30% of the training files to the testing directory"""
    amount_of_files = getAmountOffFilesInDirectory(directory)
    amount_of_files_moved = amount_of_files * 0.7 # Gets finds which number is 70% of the files
    files_counted = 0 # Counts the amount of files
    for file_name in os.listdir(directory):
        # Once loop reaches past the 70% mark it begins to move training files to the test file directory
        if files_counted >= amount_of_files_moved:
            class_name = directory.split('/')[1] # Gets name of the class I.E. Digimon or Pokemon
            shutil.move(directory + '/' + file_name, 'test/' + class_name + '/' + file_name) # Moves files
        files_counted += 1

moveFilesToTestDirectory('train/Pokemon')
moveFilesToTestDirectory('train/Digimon')
print('Amount of Pokemon training images', getAmountOffFilesInDirectory('train/Pokemon'))
print('Amount of Digimon training images', getAmountOffFilesInDirectory('train/Digimon'))
print('Amount of Pokemon test images', getAmountOffFilesInDirectory('test/Pokemon'))
print('Amount of Digimon test images', getAmountOffFilesInDirectory('test/Digimon'))

