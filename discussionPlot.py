import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



Methods = ["LR_PCA","GNB_PCA","KNN_PCA","RF_PCA","ADA_PCA","LR_LDA","GNB_LDA",
            "KNN_LDA","RF_LDA","ADA_LDA","LR", "GNB", "KNN","RF","ADA"]


Accuracy = [0.6135, 0.6225,0.5956,0.6275,0.5946,0.6345,0.6404,0.5886,0.5916,0.6325
            ,0.6135,0.6025,0.5966,0.6384,0.6135]

dat = pd.DataFrame({"Methods": Methods,
                    "Test Accuracy": Accuracy})
dat

plt.figure(figsize = (6.4,4.8))
sns.barplot(x = "Methods", y = "Test Accuracy", data = dat,
            order = dat.sort_values('Test Accuracy').Methods)