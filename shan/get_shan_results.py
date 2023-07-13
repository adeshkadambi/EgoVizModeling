import os

def dir_empty(dir):
    return len(os.listdir(dir)) == 0

# set filepath to the folder containing the frames, with each subfolder containing the frames for each video
filepath = "/workspaces/cdss-modeling/shan/videos/P-03/subclips"

# set outpath to the folder where you want to save the results
outpath = "/workspaces/cdss-modeling/shan/videos/P-03/subclips_shan"

# get list of full paths of subfolders in filepath
input_subfolders = [os.path.join(filepath, subfolder) for subfolder in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, subfolder))]

# create folder in outpath for each subfolder in filepath
for subfolder in input_subfolders:
    if not os.path.exists(os.path.join(outpath, subfolder.split("/")[-1])):
        os.mkdir(os.path.join(outpath, subfolder.split("/")[-1]))

# get zip of subfolders and corresponding output folders
infolders_outfolders = zip(input_subfolders, [os.path.join(outpath, subfolder.split("/")[-1]) for subfolder in input_subfolders]) 

# track with tqdm
for infolder, outfolder in tqdm.tqdm(infolders_outfolders):
    if not dir_empty(outfolder):
        # run model