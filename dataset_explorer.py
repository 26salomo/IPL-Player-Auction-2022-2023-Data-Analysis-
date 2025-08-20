import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import os

class dataset_editor: # input is file location
    fig_size = (12,10)
    unique_values = []
    palette = "viridis"
    def __init__(self, file_location):
        self._location = None
        self._dataframe = None
        self.location = file_location
        

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ File not found: {path}")
        try:
            self.dataframe = pd.read_excel(path)
        except Exception as e:
            raise ValueError(f"❌ Failed to read file with pandas: {e}")
        self._location = path
        print(f"✅ Loaded file: {path}")

    @property
    def dataframe(self):
        return self._dataframe

    @dataframe.setter
    def dataframe(self, new_df):
        if not isinstance(new_df, pd.DataFrame):
            raise ValueError("dataframe must be a pandas DataFrame")
        self._dataframe = new_df

    @dataframe.deleter
    def dataframe(self):
        print("🗑️ DataFrame has been deleted.")
        self._dataframe = None

    @property
    def numerical_data(self):
        return self.dataframe.select_dtypes(include=['number'])

    def categorize(self):
        obj = (self.dataframe.dtypes == 'object')
        object_cols = obj[obj].index
        print("Categorical variables:", len(object_cols))
        print("List of Categorical Column:")
        for col in object_cols:
            print(f" - {col}")

        int_ = (self.dataframe.dtypes == 'int')
        num_cols = int_[int_].index
        print("\nInteger variables:", len(num_cols))
        print("List of INT Columns:")
        for col in num_cols:
            print(f" - {col}")

        fl = (self.dataframe.dtypes == 'float')
        fl_cols = fl[fl].index
        print("\nFloat variables:", len(fl_cols))
        print("List of FLOAT Columns:")
        for col in fl_cols:
            print(f" - {col}")

    def corr_coef(self, fig_size=None):
        if fig_size is None:
            fig_size = self.fig_size
        plt.figure(figsize=fig_size)
        sns.heatmap(self.dataframe.select_dtypes(include=['number']).corr(),
                    cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
        plt.title("Correlation Coefficients")
        plt.tight_layout()
        plt.show()

    def unique_val(self, fig_size=None):
        if fig_size is None:
            fig_size = self.fig_size

        obj = (self.dataframe.dtypes == 'object')
        object_cols = obj[obj].index
        unique_values = [self.dataframe[col].nunique() for col in object_cols]

        fig, ax = subplots(figsize=fig_size)
        ax.set_xlabel("OBJECT COLUMN")
        ax.set_ylabel("UNIQUE VALUES")
        ax.set_title("Unique Values in Categorical Columns")
        plt.xticks(rotation=45)
        sns.barplot(x=object_cols, y=unique_values, ax=ax)
        for container in ax.containers:
            ax.bar_label(container, fontsize = 7)
        plt.tight_layout()
        plt.show()

    """
    cat_feature update 1:
        - visual (True/False):
        Kita bisa menampilkan dalam bentuk grafik atau hanya jumlah value tiap feature
    """
    def cat_feature(self, figsize=None, palette=None, visual = True):
        if figsize is None:
            figsize = self.fig_size
        if palette is None:
            palette = self.palette

        obj = (self.dataframe.dtypes == 'object')
        object_cols = obj[obj].index


        if visual:
            n = len(object_cols)
            if n == 0:
                print("No categorical columns to plot.")
                return

            fig, axes = subplots(nrows=1, ncols=n, figsize=figsize)
            fig.suptitle("Categorical Features Distribution")
            if n == 1:
                axes = [axes]  # Ensure it's iterable
            for i, col in enumerate(object_cols):
                counts = self.dataframe[col].value_counts()
                colors = sns.color_palette(palette, len(counts))
                sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                            ax=axes[i], palette=colors, legend=False,
                            order=counts.sort_values().index)
                axes[i].tick_params(axis='x', rotation=55)
            plt.tight_layout()
            plt.show()
        elif not visual:
            for i, col in enumerate(object_cols):
                counts = self.dataframe[col].value_counts()
                print(counts, " \n")


    @staticmethod
    def dropcol(dataframe, col_arr):
        for col in col_arr:
            if col in dataframe.columns:
                dataframe.drop(col, axis=1, inplace=True)
                print(f"✅ Column '{col}' successfully deleted")
            else:
                print(f"❌ Column '{col}' is not available")

    @staticmethod
    def fillna(dataframe, col):
        if col not in dataframe.columns:
            print(f"❌ Column '{col}' not found.")
            return
        if pd.api.types.is_numeric_dtype(dataframe[col]):
            dataframe[col] = dataframe[col].fillna(dataframe[col].mean())
            print(f"✅ Filled missing values in numeric column '{col}' with mean.")
        else:
            print(f"⚠️ Skipped non-numeric column '{col}'")

    def reset_df(self):
        self.location = self.location  # Reloads the file using location setter

    @classmethod
    def show_na(cls, dataframe, show_plot=False):
        missing_per_col = dataframe.isna().sum()
        missing_per_col = missing_per_col[missing_per_col > 0]

        if missing_per_col.empty:
            print("✅ No missing values found.")
            return missing_per_col

        total_rows = dataframe.shape[0]
        print("🧼 Missing Value Summary:\n")
        for col, count in missing_per_col.items():
            percent = (count / total_rows) * 100
            print(f" - {col}: {count} missing ({percent:.2f}%)")

        if show_plot:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=missing_per_col.index, y=missing_per_col.values,
                        hue=missing_per_col.index, palette="magma", legend=False)
            plt.ylabel("Missing Value Count")
            plt.xlabel("Columns")
            plt.title("Missing Values per Column")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        return missing_per_col