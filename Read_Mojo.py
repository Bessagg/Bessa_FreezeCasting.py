# In another terminal window, download and extract the
# latest stable h2o.jar from http://www.h2o.ai/download/
"""
cd h2o-3.36.1.2
"""

# Run the PrintMojo tool from the command line.
# This requires that graphviz is installed.

tree = 0
f"""
brew install graphviz # example for Mac OsX if not already installed
java -cp h2o.jar hex.genmodel.tools.PrintMojo --tree {tree} -i "E:\0 - UnB\PG_Freeze Casting\Bessa - FreezeCasting\mojos\Grid_GBM_Train_model_python_1662912152779_103_model_1.zip" -o model.gv -f 20 -d 3
dot -Tpng model.gv -o model.png

"""
# The image will be saved at h2o-3.36.1.2 folder as model.png


"""Code that doesnt work for me below"""

import h2o
import subprocess
from IPython.display import Image
mojo_file_name = "mojos/Grid_DRF_Train_model_python_1663115455582_1_model_1.zip"
h2o_jar_path = '/h2o-3.36.1.2/h2o.jar'
mojo_full_path = mojo_file_name
gv_file_path = "h2o-3.36.1.2/model.gv"
image_file_name = "mojos/drf_tree"


def generateTree(h2o_jar_path, mojo_full_path, gv_file_path, image_file_path, tree_id = 0):
    """Function to generate graphviz tree from the saved mojo"""
    image_file_path = image_file_path + "_" + str(tree_id) + ".png"
    result = subprocess.call(["java", "-cp", h2o_jar_path, "hex.genmodel.tools.PrintMojo", "--tree", str(tree_id), "-i", mojo_full_path , "-o", gv_file_path ], shell=False)
    result = subprocess.call(["ls",gv_file_path], shell = False)
    if result is 0:
        print("Success: Graphviz file " + gv_file_path + " is generated.")
    else:
        print("Error: Graphviz file " + gv_file_path + " could not be generated.")


def generateTreeImage(gv_file_path, image_file_path, tree_id):
    """Function to generate png from graphviz tree"""
    image_file_path = image_file_path + "_" + str(tree_id) + ".png"
    result = subprocess.call(["dot", "-Tpng", gv_file_path, "-o", image_file_path], shell=False)
    result = subprocess.call(["ls",image_file_path], shell = False)
    if result is 0:
        print("Success: Image File " + image_file_path + " is generated.")
        print("Now you can execute the follow line as-it-is to see the tree graph:")
        print("Image(filename='" + image_file_path + "\')")
    else:
        print("Error: Image file " + image_file_path + " could not be generated.")


# Just change the tree id in the function below to get which particular tree you want
tree_id = 0
generateTree(h2o_jar_path, mojo_full_path, gv_file_path, image_file_name, tree_id)
generateTreeImage(gv_file_path, image_file_name, tree_id)
