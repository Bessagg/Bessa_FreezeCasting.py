# Enables inline-plot rendering
# Utilized to create and work with dataframes
import database2dataframe
import os
import plotly.io as pio
import subprocess
import data_parser


pio.renderers.default = "browser"

"""
Possíveis tentativas:
Testar dropando todos os nulos e treinando junto (306 linhas)
Treinando separadamente (num = 464 linhas, str = 1073 linhas, str_filtered = 270 linhas)
Matrix de arvore nao tem problema com isso
Talvez usar gradientboost para aumentar os dados

"""


def generateTree(h2o_jar_path, mojo_full_path, gv_file_path, image_file_path, tree_id=0):
    image_file_path = image_file_path + "_" + str(tree_id) + ".png"
    result = subprocess.call(
        ["java", "-cp", h2o_jar_path, "hex.genmodel.tools.PrintMojo", "--tree", str(tree_id), "-i", mojo_full_path,
         "-o", gv_file_path], shell=False)
    result = subprocess.call(["ls", gv_file_path], shell=False)
    if result == 0:
        print("Success: Graphviz file " + gv_file_path + " is generated.")
    else:
        print("Error: Graphviz file " + gv_file_path + " could not be generated.")


def generateTreeImage(gv_file_path, image_file_path, tree_id):
    image_file_path = image_file_path + "_" + str(tree_id) + ".png"
    result = subprocess.call(["dot", "-Tpng", gv_file_path, "-o", image_file_path], shell=False)
    result = subprocess.call(["ls", image_file_path], shell=False)
    if result == 0:
        print("Success: Image File " + image_file_path + " is generated.")
        print("Now you can execute the follow line as-it-is to see the tree graph:")
        print("Image(filename='" + image_file_path + "\')")
    else:
        print("Error: Image file " + image_file_path + " could not be generated.")


# Load generated df
DataParser = data_parser.DataParser()
df = DataParser.load_complete_data_from_pickle()
df = df[DataParser.selected_cols_reduced]
df = DataParser.preprocess_dropna(df)
opt_save = True
seeds = [6, 18, 25, 32, 42]

# H20 DRF - Distributed Random Forest
import h2o
from h2o.estimators import H2ORandomForestEstimator

"""Distributed Random Forest (DRF) is a powerful classification and regression tool. When given a set of data, 
DRF generates a forest of classification or regression trees, rather than a single classification or regression tree. 
Each of these trees is a weak learner built on a subset of rows and columns. 
More trees will reduce the variance. 
Both classification and regression take the average prediction over all of their trees to make a final prediction, 
whether predicting for a class or numeric value. """

h2o.init()
# Split the dataset into a train and valid set:
h2o_data = h2o.H2OFrame(df, destination_frame="CatNum")
train, valid, test = h2o_data.split_frame([0.6, 0.2], seed=1234)
train.frame_id = "Train"
valid.frame_id = "Valid"
test.frame_id = "Test"

model = H2ORandomForestEstimator(model_id="DRF_Freeze_Casting", ntrees=50, max_depth=20, nfolds=10,
                                 min_rows=10
                                 )

print("Training")
X = df[df.columns.drop('porosity')].columns.values.tolist()
y = "porosity"
model.train(x=X,
            y=y,
            training_frame=train,
            validation_frame=valid)
print("Testing")
performance = model.model_performance(test_data=test)
perf = model.model_performance(valid)
print("Score History")
model.score_history()
# print("Hit ratio")
# model.hit_ratio_table(valid=True)
# Generate predictions on a validation set (if necessary):
print("Predict")
pred = model.predict(valid)
print("Importance results")
model.show()

model_file = model.download_mojo(r"E:\0 - UnB\PG_Freeze Casting\Bessa - FreezeCasting\DRF_mojo.zip",
                                 get_genmodel_jar=True)


print("Model saved to " + model_file)

h2o_jar_path = r"E:\0 - UnB\PG_Freeze Casting\Bessa - FreezeCasting\h2o-3.36.1.2\h2o.jar"
mojo_full_path = model_file
image_file_name = r"E:\0 - UnB\PG_Freeze Casting\Bessa - FreezeCasting\DRF - diagram"
gv_file_path = os.path.join(image_file_name, "drf_graph.gv")
id = 3
# generateTree(h2o_jar_path, mojo_full_path, gv_file_path, image_file_name, id)
# generateTreeImage(gv_file_path, image_file_name, id)
# Image(filename=os.path.join(image_file_name, f"drf_{id}.png"))

r"""
On terminal:
cd h2o-3.36.1.2
java -cp h2o.jar hex.genmodel.tools.PrintMojo --tree 0 -i "E:\\0 - UnB\\PG_Freeze Casting\\Bessa - FreezeCasting\\best_DRF_mojo.zip\\grid_e9a3f18f_ee2d_4c4f_9743_dd269f464100_model_4.zip" -o model.gv -f 20 -d 3

dot -Tpng model.gv -o model.png
open model.png
"""

