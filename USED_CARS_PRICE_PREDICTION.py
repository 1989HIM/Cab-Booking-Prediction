{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "USED_CARS_PRICE_PREDICTION.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyN2GXN/EPRat5bbvnX46coH",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/1989HIM/Cab-Booking-Prediction/blob/master/USED_CARS_PRICE_PREDICTION.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Fjq7H8jJ19Az",
        "outputId": "78802adf-036b-4ea3-ceee-eece254cd488"
      },
      "source": [
        "from google.colab import drive;\n",
        "drive.mount('/content/drive/')"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Drive already mounted at /content/drive/; to attempt to forcibly remount, call drive.mount(\"/content/drive/\", force_remount=True).\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1lgMEIHA2GKg"
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "%matplotlib inline\n",
        "import seaborn as sns\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "from scipy import stats\n",
        "import missingno\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor,GradientBoostingRegressor,StackingRegressor,VotingRegressor\n",
        "from sklearn.metrics import f1_score,mean_squared_error,r2_score,mean_absolute_error\n",
        "from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor\n",
        "from sklearn.preprocessing import StandardScaler,MinMaxScaler\n",
        "from sklearn.linear_model import LinearRegression,Ridge,Lasso,SGDRegressor\n",
        "import xgboost as xgb\n",
        "from sklearn.svm import SVR"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 128
        },
        "id": "CiCJMWPo2Rnp",
        "outputId": "e4fd7121-4211-4fd5-a174-6c13e8a4e7ab"
      },
      "source": [
        "data=pd.read_csv('drive/MyDrive/CAR_PRICE_PREDICTION/data.csv')\n",
        "data.head(2)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>name</th>\n",
              "      <th>year</th>\n",
              "      <th>selling_price</th>\n",
              "      <th>km_driven</th>\n",
              "      <th>fuel</th>\n",
              "      <th>seller_type</th>\n",
              "      <th>transmission</th>\n",
              "      <th>owner</th>\n",
              "      <th>mileage</th>\n",
              "      <th>engine</th>\n",
              "      <th>max_power</th>\n",
              "      <th>seats</th>\n",
              "      <th>torque</th>\n",
              "      <th>rpm</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Maruti Swift Dzire VDI</td>\n",
              "      <td>2014</td>\n",
              "      <td>450000</td>\n",
              "      <td>145500</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>First Owner</td>\n",
              "      <td>23.40</td>\n",
              "      <td>1248.0</td>\n",
              "      <td>74.00</td>\n",
              "      <td>5.0</td>\n",
              "      <td>190.0</td>\n",
              "      <td>1000.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Skoda Rapid 1.5 TDI Ambition</td>\n",
              "      <td>2014</td>\n",
              "      <td>370000</td>\n",
              "      <td>120000</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>Second Owner</td>\n",
              "      <td>21.14</td>\n",
              "      <td>1498.0</td>\n",
              "      <td>103.52</td>\n",
              "      <td>5.0</td>\n",
              "      <td>250.0</td>\n",
              "      <td>2000.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "                           name  year  selling_price  ...  seats torque     rpm\n",
              "0        Maruti Swift Dzire VDI  2014         450000  ...    5.0  190.0  1000.0\n",
              "1  Skoda Rapid 1.5 TDI Ambition  2014         370000  ...    5.0  250.0  2000.0\n",
              "\n",
              "[2 rows x 14 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 169
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QJEelO5m2UpZ",
        "outputId": "f254cfdd-a009-4b43-e204-92cdbd89eb88"
      },
      "source": [
        "data.shape"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(8128, 14)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 170
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5BSAz9cg4y45"
      },
      "source": [
        "#### <b>MISSING VALUES ANALYSIS</b>\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LWWpwiDK2fN3",
        "outputId": "529bf78b-d838-4419-8aea-2cd56a7343bf"
      },
      "source": [
        "data.replace(0,np.nan,inplace=True)\n",
        "miss_per=round(data.isna().sum()*100/len(data),2)\n",
        "print(f\"{'Feature':{15}}{'Missing %'}\\n{'='*25}\")\n",
        "print(miss_per)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Feature        Missing %\n",
            "=========================\n",
            "name             0.00\n",
            "year             0.00\n",
            "selling_price    0.00\n",
            "km_driven        0.00\n",
            "fuel             0.00\n",
            "seller_type      0.00\n",
            "transmission     0.00\n",
            "owner            0.00\n",
            "mileage          2.93\n",
            "engine           2.72\n",
            "max_power        2.72\n",
            "seats            2.72\n",
            "torque           2.73\n",
            "rpm              5.41\n",
            "dtype: float64\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 394
        },
        "id": "qgC7Tbyq8Ekj",
        "outputId": "88b6613f-ad46-4d13-b59e-f01e5bdb3a70"
      },
      "source": [
        "missingno.bar(data,log=True,figsize=(15,5),fontsize=10)\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1080x360 with 3 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 565
        },
        "id": "u1rxbDDN4cDA",
        "outputId": "22007a0f-5f14-4796-f69b-7079e5407e94"
      },
      "source": [
        "missingno.matrix(data,fontsize=14)\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1800x720 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "TytE1vNdN-aX"
      },
      "source": [
        "#### <b>Summary:-</b><br>\n",
        "- From the missing values pattern, it appears that the level of missingness is construct level which is generally referred when a case/example is partially filled. \n",
        "- Although, the column wise missing value percentage ranges from 2-5 % but row wise percentage is about 42 %, which is very high.\n",
        "- **Rpm** contains 5.41% of missing values and removing these missing values row wise will result in significant loss of data. The original dataset (i.e before cleaning) has a **torque** feature in which the values are like 190Nm @ 500 rpm.That feature was splitted into two features i.e. torque and rpm."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "A64K4ryCj3ZA"
      },
      "source": [
        "#### <b>Handling Missing Values</b>\n",
        "- **Rpm** feature can be dropped as it doesn't show any correlation with selling price.\n",
        "- For other features with missing values, we need to focus on the following question:-\n",
        "  - Is missingness low enough to delete ?\n",
        "  - Is ignoring missing data worth it ?\n",
        "  - Is the impact of missing values features,if included, significant in final prediction?\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XYD_cj6jeJVI",
        "outputId": "18911fb4-e2a5-4c01-e36f-1ad5a08034a6"
      },
      "source": [
        "data1=data.dropna()\n",
        "round((data.shape[0]-data1.shape[0])*100/data.shape[0],2)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "5.62"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 174
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6g3w6p9hq2Z5"
      },
      "source": [
        "- Removing all the missing value cases, shrinks the dataset by **5.62 %**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Jwp-eR0es7oh",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c592f6e3-015e-4e38-c6f3-40c4bd6180a8"
      },
      "source": [
        "# Splitting the dataset on the basis of valid and missing value cases\n",
        "data_temp=data.fillna(-1)\n",
        "data_valid=data_temp[data_temp['mileage']!=-1]\n",
        "data_miss=data_temp[data_temp['mileage']==-1]\n",
        "print(data_valid.shape)\n",
        "print(data_miss.shape)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "(7890, 14)\n",
            "(238, 14)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "u9U7stcssZGP"
      },
      "source": [
        "**Analyzing the difference in the distribution of selling price corresponding to valid and missing data**\n",
        "- Is the difference significant ?\n",
        "- Will removal of all the cases with missing data leads to biased results ?"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 279
        },
        "id": "18mkmx2KplgF",
        "outputId": "c231b0c4-d61c-437e-f7fc-3d1e31331db3"
      },
      "source": [
        "sns.distplot(np.log(data_valid['selling_price']),kde_kws = {'shade': True, 'linewidth': 3},label='Valid_values')\n",
        "sns.distplot(np.log(data_miss['selling_price']),kde_kws = {'shade': True, 'linewidth': 3},label='Missing_values')\n",
        "plt.yticks([])\n",
        "plt.xticks([])\n",
        "plt.title('Log (Selling_Price)')\n",
        "plt.legend()\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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PHb//ffzjW/0v9HiBRdcEF3dvHr1atraTk0dedddd3HfffclrY4333wzTz/9dNKuly4SrIUQI3bDDTewZs2aPo+tWbOGG264YcjXvvzyyxQUSFL2REmwFkKM2DXXXMNLL72E3+8H4OjRo9TV1fHEE0+wbNky5s2bx49//ON+X1tRUUFTUxMA99xzD7Nnz+b8889n//79A95v3759rFixIlo+evQoCxYsAODuu+9m+fLlzJ8/n6997Wv0t/1z/D03bdrEBRdcAEB3dzdf+cpXWLFiBUuWLOGvf/0rALt372bFihUsXryYhQsXUllZOczvUPJIsBZCjFhRURErVqxg7VojX++aNWu49tprueeee9i0aRM7duzgnXfeYceOHQNeY/PmzaxZs4Zt27bx8ssvs3HjxgHPnTNnDn6/nyNHjgDw5JNPct111wFw2223sXHjRnbt2kVPTw8vvvhiwu/jnnvu4aKLLmLDhg289dZb3HnnnXR3d/Pggw9yxx13sG3bNjZt2kR5efnQF0sRCdZCiFGJ7wrp7QJ56qmnOOecc1iyZAm7d+9mz549A77+vffe4+qrryYnJ4e8vDyuvPLKQe937bXX8uSTTwJ9g/Vbb73FypUrWbBgAW+++Sa7d+9O+D289tpr/PznP2fx4sVccMEFeL1eqqqqWLVqFT/72c/4xS9+wbFjx3A6nQlfM9kkWAshRuWqq65i3bp1bNmyBY/HQ1FREffddx/r1q1jx44dXHHFFUldDn/dddfx1FNPceDAAZRSzJo1C6/Xyze/+U2efvppdu7cya233trvPS0WC+FwGKDP81prnnnmGbZt28a2bduoqqpi7ty53HjjjTz//PM4nU5Wr17Nm2++mbT3MVwydU+I00g6ptq53W4uvPBCvvKVr3DDDTfQ0dGBy+UiPz+f+vp61q5dG+0b7s8nP/lJbr75Zr7//e8TDAZ54YUX+PrXvz7g+TNmzMBsNvOv//qv0VZ1b+AtKSmhq6uLp59+mmuuueaU11ZUVLB582Yuv/xynnnmmejjl156Kb/+9a/59a9/jVKKrVu3smTJEg4fPsz06dO5/fbbqaqqYseOHVx00UUj/E6NjrSshRCjdsMNN7B9+3ZuuOEGFi1axJIlS5gzZw433ngj55133qCvPeecc7juuutYtGgRl19+OcuXLx/yftdddx2PPfYY1157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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jB9evZEd15qO"
      },
      "source": [
        "- By visualizing the log of selling price corrosponding to valid and missing values, it appears that there is a difference in the distribution.\n",
        "- We need to determine whether this difference is significant or not by a statistical method.\n",
        "- If the difference is significant i.e. p-value < alpha, then missingness can be classified as **Missing At Random (MAR)** otherwise **Missing Completely At Random (MCAR)**."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "VtlBQDOs3Qf9"
      },
      "source": [
        "##### **Statistical Analysis for distribution:-**</br>\n",
        "- We can perform t-test to check the difference in the distribution **BUT** t-test requires an assumption that data needs to be normally distributed.\n",
        "- To check the normality assumption we can perform **Shapiro-Wilk** test in which :</br> Ho : Data is normally distributed</br> Ha : Data isn't normally distributed"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DRTj-_iE4dOW",
        "outputId": "d5b0dcf4-96e3-412d-f768-6182c67bc1fd"
      },
      "source": [
        "print(stats.shapiro(data_valid['selling_price']))\n",
        "print(stats.shapiro(data_miss['selling_price']))\n",
        "print('\\n')\n",
        "print(stats.shapiro(np.log(data_valid['selling_price'])))\n",
        "print(stats.shapiro(np.log(data_miss['selling_price'])))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "(0.5374230742454529, 0.0)\n",
            "(0.49056923389434814, 1.18300453293676e-25)\n",
            "\n",
            "\n",
            "(0.9778522253036499, 2.855246023446581e-33)\n",
            "(0.9783357381820679, 0.0010500450152903795)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "VtwR_pUw5EMz"
      },
      "source": [
        "- The p-values in both Log and without Log of data is close to zero.Thus rejecting the null hypothesis.Therefore can't perform t-test.\n",
        "- Let's try **kolmogorov-smirnov test** which checks the null hypothesis that two samples come from the same distribution"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BYoyYh6Z6UUy",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "4bcd49df-ae8b-4bb9-bb3f-1bdd7bae1da1"
      },
      "source": [
        "# kolmogorov-smirnov test \n",
        "stats.ks_2samp(data_valid['selling_price'],data_miss['selling_price'])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Ks_2sampResult(statistic=0.4674324482644769, pvalue=2.856318891836063e-44)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 178
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "bMhxfSi96cZ8"
      },
      "source": [
        "- **The pvalue is almost negligible, giving us strong evidence for rejecting the null hypothesis i.e. distributions are not from same sample** <br>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "bSuOVjDzNuOy"
      },
      "source": [
        "##### **SKEWNESS & KURTOSIS COMPARISION OF FEATURES**\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7O0eEPyNpzba",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "3025efaa-b9d8-47df-b924-5c710ff2f255"
      },
      "source": [
        "print('SKEWNESS','Valid'.rjust(13),'Missing'.rjust(13),\\\n",
        "      'KURTOSIS'.rjust(15),'Valid'.rjust(13),'Missing'.rjust(13),'\\n','='*80)\n",
        "columns=['selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'seats',\n",
        "       'torque']\n",
        "for i in columns:\n",
        "    print(f'{i:{15}}{round(data_valid[i].skew(),2):{7}}{round(data_miss[i].skew(),2):{12}}\\\n",
        "          {i:{15}}{round(data_valid[i].kurtosis(),2):{7}}{round(data_miss[i].kurtosis(),2):{12}}')"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "SKEWNESS         Valid       Missing        KURTOSIS         Valid       Missing \n",
            " ================================================================================\n",
            "selling_price     4.16        5.24          selling_price    20.69       34.75\n",
            "km_driven        11.35        0.85          km_driven       388.55        1.85\n",
            "mileage           0.07           0          mileage          -0.33           0\n",
            "engine            1.13        4.48          engine            0.73        21.0\n",
            "max_power         1.64        4.68          max_power         3.81       24.04\n",
            "seats             1.98        3.48          seats             3.78       10.32\n",
            "torque            75.3        5.87          torque         6177.22       37.32\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "e_v4ODhCQtbt"
      },
      "source": [
        "##### **Summary:**\n",
        "- Since the data is skewed, we can't perform t-test as normality assumption is violated.\n",
        "- From kolmogorov-smirnov test, we infer that removing all the cases with missing values can create some biasedness in prediction.\n",
        "- Removing all the missing value cases,even though percentage looks small, might not be beneficial.\n",
        "- Therefore, we need imputation strategies for missing values in mileage,engine,max_power,seats and torque."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "CuTlpLysYumi"
      },
      "source": [
        "# Creating a new feature years_old and Dropping name and rpm.\n",
        "import datetime\n",
        "current_date=datetime.datetime.now()\n",
        "data['years_old']=data['year'].apply(lambda x:current_date.year-x)\n",
        "data1=data.drop(['name','year','rpm'],axis=1)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "id": "5yrifMsKvYUc",
        "outputId": "7e2610dc-49d3-44d9-f8cc-0eabb49eef9a"
      },
      "source": [
        "data1.head()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>selling_price</th>\n",
              "      <th>km_driven</th>\n",
              "      <th>fuel</th>\n",
              "      <th>seller_type</th>\n",
              "      <th>transmission</th>\n",
              "      <th>owner</th>\n",
              "      <th>mileage</th>\n",
              "      <th>engine</th>\n",
              "      <th>max_power</th>\n",
              "      <th>seats</th>\n",
              "      <th>torque</th>\n",
              "      <th>years_old</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>450000</td>\n",
              "      <td>145500</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>First Owner</td>\n",
              "      <td>23.40</td>\n",
              "      <td>1248.0</td>\n",
              "      <td>74.00</td>\n",
              "      <td>5.0</td>\n",
              "      <td>190.0</td>\n",
              "      <td>7</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>370000</td>\n",
              "      <td>120000</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>Second Owner</td>\n",
              "      <td>21.14</td>\n",
              "      <td>1498.0</td>\n",
              "      <td>103.52</td>\n",
              "      <td>5.0</td>\n",
              "      <td>250.0</td>\n",
              "      <td>7</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>158000</td>\n",
              "      <td>140000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>Third Owner</td>\n",
              "      <td>17.70</td>\n",
              "      <td>1497.0</td>\n",
              "      <td>78.00</td>\n",
              "      <td>5.0</td>\n",
              "      <td>12.7</td>\n",
              "      <td>15</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>225000</td>\n",
              "      <td>127000</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>First Owner</td>\n",
              "      <td>23.00</td>\n",
              "      <td>1396.0</td>\n",
              "      <td>90.00</td>\n",
              "      <td>5.0</td>\n",
              "      <td>22.4</td>\n",
              "      <td>11</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>130000</td>\n",
              "      <td>120000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>First Owner</td>\n",
              "      <td>16.10</td>\n",
              "      <td>1298.0</td>\n",
              "      <td>88.20</td>\n",
              "      <td>5.0</td>\n",
              "      <td>11.5</td>\n",
              "      <td>14</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   selling_price  km_driven    fuel  ... seats torque years_old\n",
              "0         450000     145500  Diesel  ...   5.0  190.0         7\n",
              "1         370000     120000  Diesel  ...   5.0  250.0         7\n",
              "2         158000     140000  Petrol  ...   5.0   12.7        15\n",
              "3         225000     127000  Diesel  ...   5.0   22.4        11\n",
              "4         130000     120000  Petrol  ...   5.0   11.5        14\n",
              "\n",
              "[5 rows x 12 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 181
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fwXlrrASbWoG",
        "outputId": "200b05ec-107b-4298-a5c2-767c4892b4c8"
      },
      "source": [
        "data1.seller_type.unique()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array(['Individual', 'Dealer', 'Trustmark Dealer'], dtype=object)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 182
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "SjDPk29M67m5"
      },
      "source": [
        "#### **HEATMAP**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZspfYIik9hWw",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 325
        },
        "outputId": "9edb25f8-264f-486a-a9c4-0b9aff5ffac0"
      },
      "source": [
        "plt.figure(figsize=(15,5))\n",
        "sns.heatmap(data1.corr(),annot=True,fmt='.2g')\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1080x360 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ahJJGZhH7LmS"
      },
      "source": [
        "#### **MISSING VALUES IMPUTATION STRATEGY:**</br>\n",
        "- Missing values are present in **mileage,engine,max_power,seats,torque and rpm.**\n",
        "- First, we will split dataset into two parts i.e one with complete cases and another with incomplete cases.\n",
        "- Then Complete cases will be used for training purpose to impute the values into incomplete cases.\n",
        "- The **max_power** has strong positive correlation of 0.75 with selling price, 0.7 with engine and negative correlation with mileage and years_old.\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Ar7HWBRYhx8X",
        "outputId": "73b20fb5-e0ee-4954-a533-56646eb04e51"
      },
      "source": [
        "data_temp=data1.fillna(-1)\n",
        "data_valid=data_temp[data_temp['mileage']!=-1]\n",
        "data_miss=data_temp[data_temp['mileage']==-1]\n",
        "print(data_valid.shape)\n",
        "print(data_miss.shape)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "(7890, 12)\n",
            "(238, 12)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oY05-EXrd4s9"
      },
      "source": [
        "##### **1) Max_power**</br>\n",
        "Max power has strong positive correlation of 0.75 with **selling price** and negative correlation with **years_old**. (#these are complete features)"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VO3IbrN7p08F",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "4c9e11ae-7fbc-4c41-a6d9-7c611afba778"
      },
      "source": [
        "X=data_valid[['selling_price','years_old']]\n",
        "y=data_valid['max_power']\n",
        "rf=RandomForestRegressor(n_estimators=50)\n",
        "rf.fit(X,y)\n",
        "y_pred=rf.predict(X)\n",
        "print('R2_score =',round(rf.score(X,y),2))\n",
        "print('RMSE =',round(np.sqrt(mean_squared_error(y_pred,y)),2))\n",
        "#imputing max_power missing values\n",
        "max_pow_miss=rf.predict(data_miss[['selling_price','years_old']])\n",
        "data_miss['max_power']=max_pow_miss"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "R2_score = 0.86\n",
            "RMSE = 13.49\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "AG32vKXOi4V2"
      },
      "source": [
        "##### **2) Engine**</br>\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QumSJCSOfPRO",
        "outputId": "b9795802-9d16-4620-8fc6-6253c8737761"
      },
      "source": [
        "X=data_valid[['selling_price','max_power','km_driven']]\n",
        "y=data_valid['engine']\n",
        "rf=RandomForestClassifier(n_estimators=50)\n",
        "rf.fit(X,y)\n",
        "y_pred=rf.predict(X)\n",
        "print('R2_score =',round(rf.score(X,y),2))\n",
        "print('RMSE =',round(np.sqrt(mean_squared_error(y_pred,y)),2))\n",
        "data_miss['engine']=rf.predict(data_miss[['selling_price','max_power','km_driven']])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "R2_score = 1.0\n",
            "RMSE = 15.02\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "A-rzwcYOnXd9"
      },
      "source": [
        "#### **3) Mileage**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "R-Rre8YUkx6f",
        "outputId": "16d1fb46-28cc-425d-d579-1813fda9a363"
      },
      "source": [
        "X=data_valid[['selling_price','engine','max_power','years_old']] #not including mileage since it is missing\n",
        "y=data_valid['mileage'].astype('int')\n",
        "rf=RandomForestRegressor(n_estimators=50)\n",
        "rf.fit(X,y)\n",
        "y_pred=rf.predict(X)\n",
        "print('R2_score =',round(rf.score(X,y),2))\n",
        "print('RMSE =',round(np.sqrt(mean_squared_error(y_pred,y)),2)) \n",
        "data_miss['mileage']=rf.predict(data_miss[['selling_price','engine','max_power','years_old']])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "R2_score = 0.98\n",
            "RMSE = 0.52\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hg-ixr9ZxssQ"
      },
      "source": [
        "##### **4) Seats**\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Ha9lMW7ptf20",
        "outputId": "05ff3477-f540-41de-af3b-09e045160c84"
      },
      "source": [
        "X=data_valid[['engine','mileage','km_driven']] #not including mileage since it is missing\n",
        "y=data_valid['seats']\n",
        "rf=RandomForestClassifier(n_estimators=50)\n",
        "rf.fit(X,y)\n",
        "y_pred=rf.predict(X)\n",
        "print('R2_score =',round(rf.score(X,y),2))\n",
        "print('RMSE =',round(np.sqrt(mean_squared_error(y_pred,y)),2)) \n",
        "data_miss['seats']=rf.predict(data_miss[['engine','mileage','km_driven']]).astype('int')"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "R2_score = 0.99\n",
            "RMSE = 0.17\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5nIbimBAzcfo"
      },
      "source": [
        "##### **5) Torque**\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZaKJaZ_1zbcp",
        "outputId": "c9369237-656d-45ba-8247-9802e61cc5cb"
      },
      "source": [
        "X=data_valid[['engine','max_power']] #not including mileage since it is missing\n",
        "y=data_valid['torque']\n",
        "rf=RandomForestRegressor(n_estimators=50)\n",
        "rf.fit(X,y)\n",
        "y_pred=rf.predict(X)\n",
        "print('R2_score =',round(rf.score(X,y),2))\n",
        "print('RMSE =',round(np.sqrt(mean_squared_error(y_pred,y)),2)) \n",
        "data_miss['torque']=rf.predict(data_miss[['engine','max_power']])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "R2_score = 0.77\n",
            "RMSE = 219.85\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GVSKPmC_ZO0I"
      },
      "source": [
        "**Concatinated dataset and shuffled. All missing values imputed.**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "id": "GZY5tGmkXtKr",
        "outputId": "cc26398d-e2f8-4da8-cbd5-d7bb9a4b4587"
      },
      "source": [
        "# Concatinating dataset after imputing values\n",
        "data2=pd.concat([data_valid,data_miss],ignore_index=True)\n",
        "data2=data2.sample(frac=1).reset_index(drop=True)\n",
        "data2.head()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>selling_price</th>\n",
              "      <th>km_driven</th>\n",
              "      <th>fuel</th>\n",
              "      <th>seller_type</th>\n",
              "      <th>transmission</th>\n",
              "      <th>owner</th>\n",
              "      <th>mileage</th>\n",
              "      <th>engine</th>\n",
              "      <th>max_power</th>\n",
              "      <th>seats</th>\n",
              "      <th>torque</th>\n",
              "      <th>years_old</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>385000</td>\n",
              "      <td>120000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>First Owner</td>\n",
              "      <td>19.10</td>\n",
              "      <td>1197.0</td>\n",
              "      <td>85.8</td>\n",
              "      <td>5.0</td>\n",
              "      <td>114.0</td>\n",
              "      <td>8</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>150000</td>\n",
              "      <td>80000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>Third Owner</td>\n",
              "      <td>19.70</td>\n",
              "      <td>796.0</td>\n",
              "      <td>46.3</td>\n",
              "      <td>5.0</td>\n",
              "      <td>62.0</td>\n",
              "      <td>14</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>484999</td>\n",
              "      <td>75000</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Dealer</td>\n",
              "      <td>Manual</td>\n",
              "      <td>First Owner</td>\n",
              "      <td>19.87</td>\n",
              "      <td>1461.0</td>\n",
              "      <td>83.8</td>\n",
              "      <td>5.0</td>\n",
              "      <td>200.0</td>\n",
              "      <td>6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>400000</td>\n",
              "      <td>70000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>Second Owner</td>\n",
              "      <td>19.10</td>\n",
              "      <td>1197.0</td>\n",
              "      <td>85.8</td>\n",
              "      <td>5.0</td>\n",
              "      <td>114.0</td>\n",
              "      <td>7</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>70000</td>\n",
              "      <td>20000</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>First Owner</td>\n",
              "      <td>15.90</td>\n",
              "      <td>1527.0</td>\n",
              "      <td>57.0</td>\n",
              "      <td>5.0</td>\n",
              "      <td>96.0</td>\n",
              "      <td>18</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   selling_price  km_driven    fuel  ... seats torque years_old\n",
              "0         385000     120000  Petrol  ...   5.0  114.0         8\n",
              "1         150000      80000  Petrol  ...   5.0   62.0        14\n",
              "2         484999      75000  Diesel  ...   5.0  200.0         6\n",
              "3         400000      70000  Petrol  ...   5.0  114.0         7\n",
              "4          70000      20000  Diesel  ...   5.0   96.0        18\n",
              "\n",
              "[5 rows x 12 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 190
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9Jc2WSY_egJf"
      },
      "source": [
        "# Dropping duplicate entries\n",
        "#data2.drop_duplicates(inplace=True)\n",
        "#print(data2.shape)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MKTWC4TvZh4b"
      },
      "source": [
        "### **Outliers Analysis**</br>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 443
        },
        "id": "SmujG3JYYJT7",
        "outputId": "9059895b-0618-4021-bf0a-b617e978d475"
      },
      "source": [
        "fig, ax = plt.subplots(2, 4)\n",
        "sns.boxplot(data2['selling_price'],ax = ax[0,0])\n",
        "sns.boxplot(data2['km_driven'],ax = ax[0,1])\n",
        "sns.boxplot(data2['mileage'],ax = ax[0,2])\n",
        "sns.boxplot(data2['engine'],ax = ax[0,3])\n",
        "sns.boxplot(data2['max_power'],ax = ax[1,0])\n",
        "sns.boxplot(data2['seats'],ax = ax[1,1])\n",
        "sns.boxplot(data2['torque'],ax = ax[1,2])\n",
        "sns.boxplot(data2['years_old'],ax = ax[1,3])\n",
        "fig.set_figheight(7)\n",
        "fig.set_figwidth(22)\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAABN0AAAGqCAYAAAA/TTJVAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzde5xlZXkn+t9DtwSEJCoaj4OX1mmNkHTUQGJuJ9MwYEA0TBJj8JgDxkvEjEjUGcdIj91kGs/JGDXYZqLiIUAkauJoUG4JCI4zmWgCo4iCl4rgpceoASEBUWl4zx9773JXdV12Va2qXZfv9/OpT9Vee621n3ddnrXWU+9au1prAQAAAAC6c8C4AwAAAACA9UbRDQAAAAA6pugGAAAAAB1TdAMAAACAjim6AQAAAEDHNi9k5Ic+9KFty5YtyxQKsF5cf/31/9hae9i44+iS/AeMQv4DNir5D9io5sp/Cyq6bdmyJdddd103UQHrVlV9cdwxdE3+A0Yh/wEblfwHbFRz5T+3lwIAAABAxxTdAAAAAKBjim4AAAAA0DFFNwAAAADomKIbAAAAAHRM0Q0AAAAAOqboBgAAAAAdU3QDAAAAgI4pugEAAABAxxTdAAAAAKBjim4AAAAA0DFFNwAAAADomKIbAAAAAHRM0Q0AAAAAOqboBgAAAAAd27ySH7Znz55MTExMGbZ3794kyeGHH77f+Fu3bs0ZZ5yxIrEBLJdB7psp38lzAGvPTOe0CzXXOXAXHF+AUU3Pacudn2YiZ7FerWjRbWJiIp/41M2574EPmRy26Vt3Jkn+4TtTQ9n0rdtXMjSAZTPIfUlL8r18J88BrE0zndMu1GznwF1wfAEWYnpOW878NBM5i/VsRYtuSXLfAx+Se5749MnXB3/m8iSZMmx4OMB6MHxhNsh38hzA2jX9nHahZjsH7oLjC7BQwzltOfPTTOQs1jPPdAMAAACAjim6AQAAAEDHFN0AAAAAoGOKbgAAAADQMUU3AAAAAOiYohsAAAAAdEzRDQAAAAA6pugGAAAAAB1TdAMAAACAjim6AQAAAEDHFN0AAAAAoGOKbgAAAADQMUU3AAAAAOiYohsAAAAAdEzRDQAAAAA6pugGAAAAAB1TdAMAAACAjim6AQAAAEDHFN0AAAAAoGOKbgAAAADQMUU3AAAAAOiYohsAAAAAdEzRDQAAAAA6pugGAAAAAB1TdAMAAACAjim6AQAAAEDHFN0AAAAAoGOKbgAAAADQMUU3AAAAAOiYohsAAAAAdEzRDQAAAAA6tmxFtz179mTPnj3LNftV+9kAXecgOQ1YK+QrxsF2x2phW9xYrG9GsXm5ZjwxMbFcs17Vnw3QdQ6S04C1Qr5iHGx3rBa2xY3F+mYUbi8FAAAAgI4pugEAAABAxxTdAAAAAKBjim4AAAAA0DFFNwAAAADomKIbAAAAAHRM0Q0AAAAAOqboBgAAAAAdU3QDAAAAgI4pugEAAABAxxTdAAAAAKBjim4AAAAA0DFFNwAAAADomKIbAAAAAHRM0Q0AAAAAOqboBgAAAAAdU3QDAAAAgI4pugEAAABAxxTdAAAAAKBjim4AAAAA0DFFNwAAAADomKIbAAAAAHRM0Q0AAAAAOqboBgAAAAAdU3QDAAAAgI4pugEAAABAxxTdAAAAAKBjim4AAAAA0DFFNwAAAADomKIbAAAAAHRs87gDWA633357vvzlL2f79u3zjrtp06bcd999I833QQ96UO688868/OUvz9ve9racffbZOf/883PTTTfNO78DDzww9957b0488cRcfvnlk8Nf+cpX5ogjjsiZZ56Zu+++e3L4ox/96LzsZS/Lf/yP/zGttbzqVa/KG97whpx77rl58IMfnLPPPjs7d+7MN7/5zbz0pS/Nt7/97STJ5s2bc+SRR+bwww/PFVdckV/91V/NgQcemIsvvjinnnpqnv/858/avmuuuSa/+7u/mxe/+MV55zvfmXPPPTdbt24dadlMX9Y7d+7MMccckyQ57rjjsm/fvsn3DjjggFxzzTVTxp+YmMhLX/rSfOc738lrX/vavP/978+2bdty8cUX5/TTT88pp5wyOe5tt9022f7DDjtsynzmem8US51+vbE8VofPfvaz+fa3vz1vTtu8efPkvvbUpz41X/va1/KlL30pr3/967Nly5bs2LEjd999d7785S/n93//93PUUUcl6a3nQa7ZvXt3DjvssMl1f9ppp+W1r33trPnANjI+lv3ys4xh/Rrs3y972cvy5je/eXI/H97vb7nllrzqVa/K61//+slj5kYh/8Hopp+jP+UpT8nnPve5ea+nX/jCF2ZiYiJJcvrpp+fCCy/MPffck0MOOSSXXXbZ5H64devWvO9975txHh/+8Icn/56YmMiLXvSitNaybdu27NmzZ8q+/I53vCNXXHFFnvnMZ+aVr3xlkuSYY45Jay1VlW3btk3Z50fJAyeccMJkLWLYK1/5yjzzmc+cte2zueSSS/KmN71p0dOPYr52XXzxxTnvvPP2q0Ms1rrs6fblL3955HFHLbglyR133JHWWt70pjfl7rvvzq5du6YU3Oaa33e/+9201qYU3JLkjW98Y3bv3j2l4JYkX/rSl7Jr167cc889+fa3v53Xve51ufvuu7N79+5ceOGFufHGG3PRRRdl9+7dUzbyffv25cYbb8wVV1yRJPnzP//zXHzxxUmSiy66aM72ve51r0uSvO1tb5v8rMU655xzpsQ07P77799v/EE7Wms555xzcuONN07G/da3vnXKuMPtn26u90ax1OnXG8tjdZjpQDaT4X3tYx/7WG699dbcf//92blzZy688MLcfPPN+dKXvpTWWnbu3Dk57oUXXpibbropN9988+S6Hqz7nTt3zpkPbCPjY9kvP8sY1q/B/r179+4p+/nwfr9r167J4+hGI//B4n384x8f6Xp6UHBLete899xzT5JM1gYG++FsBbfpdu/endZakuTGG2+cMo+LLrposkbwwQ9+cHKawfittf32+VHywGzXKW984xtHinm6P/iDP1jS9KOYr13nnXdekv3rEIu17opu03tQLYfBhnnXXXd1Mq9bb711xveG5z+4mL711ltz+eWXTxbwZpp2EN9Mzj///BmHX3PNNfsVx2699dYpiWA2M/W+2bdvX6699tocd9xxM05z7LHHTv49MTExpR379u3brw3vfve7k/Sq0ldeeWVaa7nyyitz2223TY4z13ujWOr0643lsTq85CUvWfI87rrrrlx22WX7Dbv++usn1/PAFVdckYmJicl1P8hDM+UD28j4WPbLzzKG9Wt4/7711lsn9/Ph499ll102eQwcHDM3CvkPRnfDDTfM+t5c19MvfOEL55zviSeeOLkfzmVwLT79mjpJXvziF0/O45JLLpny3hve8IbJO9MGhvf5UfLACSecMGtcrbUpxb1RXHLJJVOKgAudfhTztWvQ8WdgUIdYimW7vXTv3r255557cuaZZ04Om5iYyAHfnXujGTjg2/+UiYl/njL9KOba6NeLQXHs3nvvXfC0F1100Yy3mA56uU23e/fuXHDBBQv+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AOiYohsAAAAAdEzRDQAAAAA6tnncAYzbnj17MjExMev7e/fuTZIcfvjhI89z69atOeOMM5YcG8BM9uzZkyTyDLAqvfCFL8wdd9yRf/Wv/pU8BbBI812nzmQx165L4boX5rfhi24TExP5xKduzn0PfMiM72/61p1Jkn/4zmiLatO3bu8sNoCZXHnllUkU3YDV6atf/WruvvvuBV8sAvA9812nzmSh165L4boXRrPhi25Jct8DH5J7nvj0Gd87+DOXJ8ms7882PgAAACzWXNepM1notetSuO6F0XimGwAAAAB0TNENAAAAADqm6AYAAAAAHVN0AwAAAICOKboBAAAAQMcU3QAAAACgY4puAAAAANAxRTcAAAAA6JiiGwAAAAB0TNENAAAAADqm6AYAAAAAHVN0AwAAAICOKboBAAAAQMcU3QAAAACgY4puAAAAANAxRTcAAAAA6JiiGwAAAAB0TNENAAAAADqm6AYAAAAAHVN0AwAAAICOKboBAAAAQMcU3QAAAACgY4puAAAAANAxRTcAAAAA6JiiGwAAAAB0TNENAAAAADqm6AYAAAAAHVN0AwAAAICOKboBAAAAQMcU3QAAAACgY4puAAAAANAxRTcAAAAA6NiaKbrt2bMne/bsGXcYq5plBBvDt771rXzrW98adxgA+9mzZ0++853vJEn27t3rvATYUFyPrU/WK0uxedwBjGpiYmLcIax6lhFsDK21cYcAMKOJiYncf//9SZJ77rnHuQmwoch565P1ylKsmZ5uAAAAALBWKLoBAAAAQMcU3QAAAACgY4puAAAAANAxRTcAAAAA6JiiGwAAAAB0TNENAAAAADqm6AYAAAAAHVN0AwAAAICOKboBAAAAQMcU3QAAAACgY4puAAAAANAxRTcAAAAA6JiiGwAAAAB0TNENAAAAADqm6AYAAAAAHVN0AwAAAICOKboBAAAAQMcU3QAAAACgY4puAAAAANAxRTcAAAAA6JiiGwAAAAB0TNENAAAAADqm6AYAAAAAHVN0AwAAAICOKboBAAAAQMcU3QAAAACgY4puAAAAANAxRTcAAAAA6NiyFd1uu+22vOxlL8ttt922XB/BDG644YZs37591p+nPe1pc74/3895552X7du35/nPf362b9+ek08+Ob/xG7+R4447Ltu3b8/5558/JZ7haY855phce+21ednLXpaJiYn81m/9Vl7ykpfst41cd911OfbYY3P99dfnNa95TbZv357Xvva1s7b54osvzvbt2/OWt7wlJ510Ut785jdPxjLYBq+55pps3749p5122pTPG0z7nOc8x7Y6A/vx4gxv92vVcrfB/Of37Gc/ezI/MR5y4NLcfffdU85Lrr/++nGHNFa2J9YS2yvM7+yzz8727dtzzjnn7PfeySefnO3bt+eXfumXpgx/znOek+3bt+fXf/3XR57fXNfEw9fOwyYmJnLSSSdlYmJi5GkWa73li67bs2xFtwsvvDA33nhjLrroouX6CBbhu9/97pKmv/jii5MkX/jCF5Ikd955Z2655Zbs27cvSeZc3621nHPOObnxxhuze/fu3HTTTbn55pv3m2bXrl25//77s3PnzvzP//k/kyQf+chHZp3veeedlyR573vfm7vvvjvve9/7JmMZbIOve93rkiRf/OIXp3zeYNqvfvWrttUZ2I9hfL7+9a8n6eUnxkMO7NbOnTvHHcJY2Z5YS2yvML9rr702SXLVVVft996dd96ZJPnmN785ZfjgvO4rX/nKyPOb65p4+Np52O7du3P33Xdn9+7dI0+zWOstX3TdnmUput1222258sor01rLlVdeuW4qnqvdDTfcMO4QkmSyt9tMPUD27duX1lpuvfXWyWFXXHHF5DZy3XXX5a677kqSyd8DM1X2B0XA2bTWcumll04WBZPksssuy2233bbftJdeeqltdYj9eHGmb/drsbfbcrfB/Of37Gc/e8prvd1WnhzYvbvuumvD9nazPbGW2F5hfmefffaU18O9004++eQp7w16u00/nxvu7Tbb/F7zmtdMGT58TTz92nlwjJ2YmJi83r711lun9HabbZrFWm/5Yjnas7mDuPZz4YUX5v7770+S3Hfffbnooovy8pe/fEnz3Lt3b+65556ceeaZXYQ4aWJiIgd8t3U2vwO+/U+ZmPjnzuNcSy666KI8//nPH3n8e++9d3Ib2bVr16zjzVTZH/RUm8t999035fW+ffty0UUX5ZJLLtlvvC621fViOfZjYDSDXm4DerutPDlwcfbu3Tu53Gayc+fOXHrppSsY0epge2Itsb0uXpfXrF1fp3ZtI133TkxM5OCDD54ybNArbeCqq67KWWedleR7vdwGBr3dpp/PDfd2m21+g15uA8PXxNOvnQfH2Om923bv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\n",
            "text/plain": [
              "<Figure size 1584x504 with 8 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 97
        },
        "id": "fBlbl-7DYK3S",
        "outputId": "1b731efa-f8c5-4785-9c18-5c588cd17597"
      },
      "source": [
        "data[data['selling_price']==data.selling_price.max()]"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>name</th>\n",
              "      <th>year</th>\n",
              "      <th>selling_price</th>\n",
              "      <th>km_driven</th>\n",
              "      <th>fuel</th>\n",
              "      <th>seller_type</th>\n",
              "      <th>transmission</th>\n",
              "      <th>owner</th>\n",
              "      <th>mileage</th>\n",
              "      <th>engine</th>\n",
              "      <th>max_power</th>\n",
              "      <th>seats</th>\n",
              "      <th>torque</th>\n",
              "      <th>rpm</th>\n",
              "      <th>years_old</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>170</th>\n",
              "      <td>Volvo XC90 T8 Excellence BSIV</td>\n",
              "      <td>2017</td>\n",
              "      <td>10000000</td>\n",
              "      <td>30000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Automatic</td>\n",
              "      <td>First Owner</td>\n",
              "      <td>42.0</td>\n",
              "      <td>1969.0</td>\n",
              "      <td>400.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>640.0</td>\n",
              "      <td>870.0</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "                              name  year  ...    rpm  years_old\n",
              "170  Volvo XC90 T8 Excellence BSIV  2017  ...  870.0          4\n",
              "\n",
              "[1 rows x 15 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 193
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "EcSllMvZHSkM"
      },
      "source": [
        "The actual price of brand new <b>\"Volvo XC90 T8 Excellence BSIV\"</b> is about 1.31 crore, so this selling price looks valid. Although, from the box plot it appears as an outlier but removing a valid outlier doesn't feels like a good idea."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "h25RzMNQPV12",
        "outputId": "43eee8b5-9a8d-4c85-80d7-f66a4706b976"
      },
      "source": [
        "q1,q3=np.percentile(data2['km_driven'].values,[25,75])\n",
        "upper_fence=q3+1.5*(q3-q1)\n",
        "lower_fence=q1-1.5*(q3-q1)\n",
        "x=data2[data2['km_driven']>upper_fence]\n",
        "print('Pearson correlation=',stats.pearsonr(x['km_driven'],x['selling_price'])[0])\n",
        "data2['km_driven'][data2['km_driven']>upper_fence]=upper_fence"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Pearson correlation= 0.08071010575941727\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6PncZQEnaotB"
      },
      "source": [
        "Since pearson correlation is negligible, the outliers are assigned with upper_fence value."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 266
        },
        "id": "Cu8p6UZCY8X0",
        "outputId": "606aa1bb-e2aa-4336-c2ac-57b16f35bee9"
      },
      "source": [
        "q1,q3=np.percentile(data2['mileage'].values,[25,75])\n",
        "upper_fence=q3+1.5*(q3-q1)\n",
        "lower_fence=q1-1.5*(q3-q1)\n",
        "x=data2[data2['mileage']>upper_fence]\n",
        "x.sort_values(by='mileage',ascending=False)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>selling_price</th>\n",
              "      <th>km_driven</th>\n",
              "      <th>fuel</th>\n",
              "      <th>seller_type</th>\n",
              "      <th>transmission</th>\n",
              "      <th>owner</th>\n",
              "      <th>mileage</th>\n",
              "      <th>engine</th>\n",
              "      <th>max_power</th>\n",
              "      <th>seats</th>\n",
              "      <th>torque</th>\n",
              "      <th>years_old</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>3748</th>\n",
              "      <td>10000000</td>\n",
              "      <td>30000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Automatic</td>\n",
              "      <td>First Owner</td>\n",
              "      <td>42.00</td>\n",
              "      <td>1969.0</td>\n",
              "      <td>400.00</td>\n",
              "      <td>4.0</td>\n",
              "      <td>640.0</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5192</th>\n",
              "      <td>330000</td>\n",
              "      <td>10000</td>\n",
              "      <td>CNG</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>Second Owner</td>\n",
              "      <td>33.44</td>\n",
              "      <td>796.0</td>\n",
              "      <td>40.30</td>\n",
              "      <td>4.0</td>\n",
              "      <td>60.0</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5568</th>\n",
              "      <td>330000</td>\n",
              "      <td>10000</td>\n",
              "      <td>CNG</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>Second Owner</td>\n",
              "      <td>33.44</td>\n",
              "      <td>796.0</td>\n",
              "      <td>40.30</td>\n",
              "      <td>4.0</td>\n",
              "      <td>60.0</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6338</th>\n",
              "      <td>260000</td>\n",
              "      <td>67000</td>\n",
              "      <td>CNG</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>First Owner</td>\n",
              "      <td>33.44</td>\n",
              "      <td>796.0</td>\n",
              "      <td>40.30</td>\n",
              "      <td>4.0</td>\n",
              "      <td>60.0</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1569</th>\n",
              "      <td>370000</td>\n",
              "      <td>16000</td>\n",
              "      <td>CNG</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>First Owner</td>\n",
              "      <td>33.00</td>\n",
              "      <td>796.0</td>\n",
              "      <td>47.30</td>\n",
              "      <td>5.0</td>\n",
              "      <td>69.0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1128</th>\n",
              "      <td>270000</td>\n",
              "      <td>80000</td>\n",
              "      <td>CNG</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>Second Owner</td>\n",
              "      <td>32.52</td>\n",
              "      <td>998.0</td>\n",
              "      <td>58.33</td>\n",
              "      <td>5.0</td>\n",
              "      <td>78.0</td>\n",
              "      <td>8</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8063</th>\n",
              "      <td>430000</td>\n",
              "      <td>20000</td>\n",
              "      <td>CNG</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>First Owner</td>\n",
              "      <td>32.26</td>\n",
              "      <td>998.0</td>\n",
              "      <td>58.30</td>\n",
              "      <td>4.0</td>\n",
              "      <td>78.0</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "      selling_price  km_driven    fuel  ... seats torque years_old\n",
              "3748       10000000      30000  Petrol  ...   4.0  640.0         4\n",
              "5192         330000      10000     CNG  ...   4.0   60.0         2\n",
              "5568         330000      10000     CNG  ...   4.0   60.0         2\n",
              "6338         260000      67000     CNG  ...   4.0   60.0         4\n",
              "1569         370000      16000     CNG  ...   5.0   69.0         1\n",
              "1128         270000      80000     CNG  ...   5.0   78.0         8\n",
              "8063         430000      20000     CNG  ...   4.0   78.0         2\n",
              "\n",
              "[7 rows x 12 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 195
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "q8SsAEIrcoNZ"
      },
      "source": [
        "One thing is common here, that all the vehicles,having high mileage, run on CNG fuel. The highest value i.e 42.0 kmpl is correct and verified by internet search. All these values will be kept as it is."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "id": "hEr9Rbj3bVZl",
        "outputId": "f0ee1527-161d-41d9-eac5-c5d5a1d7a3d2"
      },
      "source": [
        "q1,q3=np.percentile(data2['engine'].values,[25,75])\n",
        "upper_fence=q3+1.5*(q3-q1)\n",
        "lower_fence=q1-1.5*(q3-q1)\n",
        "x=data2[data2['engine']>upper_fence]\n",
        "data2.sort_values(by='engine',ascending=False).head()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>selling_price</th>\n",
              "      <th>km_driven</th>\n",
              "      <th>fuel</th>\n",
              "      <th>seller_type</th>\n",
              "      <th>transmission</th>\n",
              "      <th>owner</th>\n",
              "      <th>mileage</th>\n",
              "      <th>engine</th>\n",
              "      <th>max_power</th>\n",
              "      <th>seats</th>\n",
              "      <th>torque</th>\n",
              "      <th>years_old</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>181</th>\n",
              "      <td>4100000</td>\n",
              "      <td>17000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Automatic</td>\n",
              "      <td>First Owner</td>\n",
              "      <td>9.5</td>\n",
              "      <td>3604.0</td>\n",
              "      <td>280.0</td>\n",
              "      <td>5.0</td>\n",
              "      <td>347.0</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4026</th>\n",
              "      <td>4100000</td>\n",
              "      <td>17000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Automatic</td>\n",
              "      <td>First Owner</td>\n",
              "      <td>9.5</td>\n",
              "      <td>3604.0</td>\n",
              "      <td>280.0</td>\n",
              "      <td>5.0</td>\n",
              "      <td>347.0</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6215</th>\n",
              "      <td>4100000</td>\n",
              "      <td>17000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Automatic</td>\n",
              "      <td>First Owner</td>\n",
              "      <td>9.5</td>\n",
              "      <td>3604.0</td>\n",
              "      <td>280.0</td>\n",
              "      <td>5.0</td>\n",
              "      <td>347.0</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4492</th>\n",
              "      <td>4100000</td>\n",
              "      <td>17000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Automatic</td>\n",
              "      <td>First Owner</td>\n",
              "      <td>9.5</td>\n",
              "      <td>3604.0</td>\n",
              "      <td>280.0</td>\n",
              "      <td>5.0</td>\n",
              "      <td>347.0</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7531</th>\n",
              "      <td>4100000</td>\n",
              "      <td>17000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Automatic</td>\n",
              "      <td>First Owner</td>\n",
              "      <td>9.5</td>\n",
              "      <td>3604.0</td>\n",
              "      <td>280.0</td>\n",
              "      <td>5.0</td>\n",
              "      <td>347.0</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "      selling_price  km_driven    fuel  ... seats torque years_old\n",
              "181         4100000      17000  Petrol  ...   5.0  347.0         4\n",
              "4026        4100000      17000  Petrol  ...   5.0  347.0         4\n",
              "6215        4100000      17000  Petrol  ...   5.0  347.0         4\n",
              "4492        4100000      17000  Petrol  ...   5.0  347.0         4\n",
              "7531        4100000      17000  Petrol  ...   5.0  347.0         4\n",
              "\n",
              "[5 rows x 12 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 196
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "up_dHBc0SWIi"
      },
      "source": [
        "Outlier values in **engine, max_power and seats** are valid values. These values will not be removed"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 213
        },
        "id": "t0eODfDgPuto",
        "outputId": "76fa3e13-2019-402c-ce50-cc1cc25acacf"
      },
      "source": [
        "data.sort_values(by='torque',ascending=False).head(2)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>name</th>\n",
              "      <th>year</th>\n",
              "      <th>selling_price</th>\n",
              "      <th>km_driven</th>\n",
              "      <th>fuel</th>\n",
              "      <th>seller_type</th>\n",
              "      <th>transmission</th>\n",
              "      <th>owner</th>\n",
              "      <th>mileage</th>\n",
              "      <th>engine</th>\n",
              "      <th>max_power</th>\n",
              "      <th>seats</th>\n",
              "      <th>torque</th>\n",
              "      <th>rpm</th>\n",
              "      <th>years_old</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>778</th>\n",
              "      <td>Ford Endeavour Hurricane Limited Edition</td>\n",
              "      <td>2013</td>\n",
              "      <td>1075000</td>\n",
              "      <td>110000</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Automatic</td>\n",
              "      <td>Third Owner</td>\n",
              "      <td>12.8</td>\n",
              "      <td>2953.0</td>\n",
              "      <td>156.0</td>\n",
              "      <td>7.0</td>\n",
              "      <td>38038.7</td>\n",
              "      <td>1250.0</td>\n",
              "      <td>8</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2000</th>\n",
              "      <td>Honda Jazz Select Edition Active</td>\n",
              "      <td>2011</td>\n",
              "      <td>350000</td>\n",
              "      <td>80000</td>\n",
              "      <td>Petrol</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>Second Owner</td>\n",
              "      <td>16.0</td>\n",
              "      <td>1198.0</td>\n",
              "      <td>90.0</td>\n",
              "      <td>5.0</td>\n",
              "      <td>11011.2</td>\n",
              "      <td>2400.0</td>\n",
              "      <td>10</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "                                          name  year  ...     rpm  years_old\n",
              "778   Ford Endeavour Hurricane Limited Edition  2013  ...  1250.0          8\n",
              "2000          Honda Jazz Select Edition Active  2011  ...  2400.0         10\n",
              "\n",
              "[2 rows x 15 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 197
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MZA79y8gV2cT"
      },
      "source": [
        "The actual torque of **Ford Endeavour** is **380Nm @ 2500rpm** and **Honda Jazz 110 (11.2) @ 4800**. So, imputing the correct values."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TdWN5qZEWfCx"
      },
      "source": [
        "data2['torque'][data2['torque']==38038.7]=380\n",
        "data2['torque'][data2['torque']==11011.2]=110\n",
        "data2['torque'][data2['torque']==-1]=58.560\n",
        "data2['max_power'][data2['max_power']==-1]=38.840"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 111
        },
        "id": "DuMwlCtcaE4r",
        "outputId": "490f21df-d23e-4834-9632-857d734b6f96"
      },
      "source": [
        "data2.sort_values(by='years_old',ascending=False).head(2)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>selling_price</th>\n",
              "      <th>km_driven</th>\n",
              "      <th>fuel</th>\n",
              "      <th>seller_type</th>\n",
              "      <th>transmission</th>\n",
              "      <th>owner</th>\n",
              "      <th>mileage</th>\n",
              "      <th>engine</th>\n",
              "      <th>max_power</th>\n",
              "      <th>seats</th>\n",
              "      <th>torque</th>\n",
              "      <th>years_old</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2239</th>\n",
              "      <td>300000</td>\n",
              "      <td>10000</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>Third Owner</td>\n",
              "      <td>15.00</td>\n",
              "      <td>998.0</td>\n",
              "      <td>96.588</td>\n",
              "      <td>5.0</td>\n",
              "      <td>149.50</td>\n",
              "      <td>38</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4133</th>\n",
              "      <td>55000</td>\n",
              "      <td>120000</td>\n",
              "      <td>LPG</td>\n",
              "      <td>Individual</td>\n",
              "      <td>Manual</td>\n",
              "      <td>Second Owner</td>\n",
              "      <td>15.96</td>\n",
              "      <td>796.0</td>\n",
              "      <td>41.140</td>\n",
              "      <td>4.0</td>\n",
              "      <td>60.12</td>\n",
              "      <td>30</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "      selling_price  km_driven    fuel  ... seats  torque years_old\n",
              "2239         300000      10000  Diesel  ...   5.0  149.50        38\n",
              "4133          55000     120000     LPG  ...   4.0   60.12        30\n",
              "\n",
              "[2 rows x 12 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 199
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 334
        },
        "id": "D83emzErXxCT",
        "outputId": "7fa80007-01a7-4c64-a465-58f8a7c658e8"
      },
      "source": [
        "fig,ax=plt.subplots(1,2)\n",
        "sns.barplot(x='fuel',y='selling_price',hue='fuel',data=data2,ax=ax[0])\n",
        "sns.countplot(x='fuel',hue='fuel',data=data2,ax=ax[1])\n",
        "fig.set_figwidth(15)\n",
        "fig.set_figheight(5)\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1080x360 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 346
        },
        "id": "-xR0J81kedBU",
        "outputId": "469e3d0c-7296-477b-e67b-7ecce5029557"
      },
      "source": [
        "fig,ax=plt.subplots(1,2)\n",
        "sns.barplot(x='seller_type',y='selling_price',hue='seller_type',data=data2,ax=ax[0])\n",
        "sns.countplot(x='seller_type',data=data2,ax=ax[1],dodge=True)\n",
        "fig.set_figwidth(15)\n",
        "fig.set_figheight(5)\n",
        "fig.set\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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tSt33XUXzz///DptjRgxgm984xuceOKJ9OjRg/nz59O1a9cKXJXaSmY+QulZsGs7pJG6CXyhiXbGA+PbNjpJUq0zyZOqwFf2ebXSIQj4v//7PwYPHszy5cvp0qULn/70p/nyl78MwGc/+1nmzJnD0KFDyUzq6ur4/e9/z4knnsjHPvYx9t57b+rr69l9993Xafewww7jqaee4sADDwRKj1b41a9+RefOndv1+iRJUm0wyZNUlVryyIO2tnLlyiaPderUie985zt85zvfWefY/fff3+g5r7/++prts846i7POOmudOo8//vgGRCpJkjZlPkJBkiRJkmqISZ4kSZIk1RCTPEmSJEmqISZ5kiRJklRDTPIkSZIkqYaY5EmSJElSDTHJk6T1WLx4MYMHD2bw4MHsuOOO9O7de83+22+/vUFt/vCHP+TNN99s40hL5syZw1577bXeOltssQVDhgxhjz32YNiwYUyYMKGsrylJktqHz8mTVJUOuvKgNm3v3jPvbfLYdtttxyOPPALAhRdeSI8ePfjKV76y5viKFSvo0qV1f05/+MMfctJJJ7HllltuWMBNWLFiRYvrvuc97+Hhhx8GYPbs2Rx99NFkJqeeemqbxtSYDfk3kyRJLeOdPEnaAKNHj+b0009n//3355xzzuHCCy/k+9///prje+21F3PmzOGNN97gox/9KIMGDWKvvfbihhtu4IorruDFF1/kQx/6EB/60IcA6NGjB2PHjmXgwIF8+MMf5qGHHmL48OHsuuuuTJ48GSjdLfvABz7A0KFDGTp0KPfddx8Ad999Nx/4wAcYNWoUe+655zvinD17NkOGDGHq1KnNXs+uu+7KZZddxhVXXAHAG2+8wWc+8xmGDRvGkCFDuPnmm5uNoaGVK1cyduxY9ttvP/bZZx9+9rOfrTdOSZLUdvwYVZI20Lx587jvvvvo3LkzF154YaN1br/9dnbeeWduvfVWAJYuXUrPnj257LLLuOuuu9h+++2BUlJ18MEHc+mll/Lxj3+cr3/960yZMoUnn3ySU045hVGjRtGrVy+mTJlCt27dmDlzJieccALTpk0DYMaMGTz++OP079+fOXPmAPDMM89w/PHHM2HCBAYNGrTe6xk6dChPP/00AN/+9rc5+OCDGT9+PK+++irDhg3jwx/+cLMxrPbzn/+cnj17MnXqVN566y0OOuggDjvssHXilCRJ5WGSJ0kb6Nhjj6Vz587N1tl77705++yzOffcczniiCP4wAc+0Gi9zTbbjJEjR645Z/PNN6dr167svffea5K25cuXc8YZZ/DII4/QuXNnnn322TXnDxs27B2J06JFizjyyCO56aabWnzXLDPXbN95551Mnjx5zd3JZcuW8cILL7Dzzjs3GUPDcx999FEmTZoElBLbmTNnstlmm60TpyRJansmeZK0gbp3775mu0uXLqxatWrN/rJlywDYbbfdmDFjBrfddhtf//rXOeSQQzj//PPXaatr165EBACdOnVi8803X7O9ep7d5Zdfzg477MBf/vIXVq1aRbdu3RqNBaBnz5707duXP//5zy1O8h5++GH22GMPoJTw/fa3v+W9733vO+pceOGFTcawWmZy5ZVXMmLEiHeU33333evEKUmS2p5z8iSpDfTr148ZM2YApSGJzz33HAAvvvgiW265JSeddBJjx45dU2errbbi73//e6teY+nSpey000506tSJX/7yl6xcubLJupttthm/+93vmDhxItdee+16254zZw5f+cpXOPPMMwEYMWIEV1555Zq7e6sXaGlJDCNGjODqq69m+fLlADz77LO88cYbrbpWSZK04byTJ0lt4JhjjmHixIkMHDiQ/fffn9122w2Axx57jLFjx9KpUye6du3K1VdfDcCYMWMYOXIkO++8M3fddVeLXuPzn//8mtcZOXLkeu+Kde/enVtuuYVDDz2UHj16MGrUqHcc/+tf/8qQIUNYtmwZW221FV/84hcZPXo0AN/4xjf40pe+xD777MOqVavo378/t9xyS4ti+OxnP8ucOXMYOnQomUldXR2///3vW3SNkiRp40XDORjVor6+Ptee6C91ZC9ctHelQwCg7/mPVTqEDfbUU0+tGUqo2tXY+xwR0zOzvkIhVZ3W9pH7jp1Yxmi0oaZfenKlQ5DUwTXXPzpcU5IkSZJqiEmeJEmSJNUQkzxJkiRJqiFlTfIiYnxELIyIx5s4fmJEPBoRj0XEfRGx/qf1StpkVeMcYrWc768kSW2j3HfyJgAjmzn+HPDBzNwb+BYwrszxSKpS3bp1Y/HixSYCNSozWbx4caPP3ZMkSa1T1kcoZOY9EdGvmeP3Ndh9AOhTzngkVa8+ffowb948Fi1aVOlQVCbdunWjTx+7AUmSNlZHek7eacB/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\n",
            "text/plain": [
              "<Figure size 1080x360 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 345
        },
        "id": "SRFS6yXy9xJ4",
        "outputId": "62fbb710-29fe-4897-c6f7-a94ff4d53555"
      },
      "source": [
        "fig,ax=plt.subplots(1,2)\n",
        "sns.barplot(x='transmission',y='selling_price',hue='fuel',data=data2,ax=ax[0])\n",
        "sns.countplot(x='transmission',hue='fuel',data=data2,ax=ax[1])\n",
        "fig.set_figwidth(15)\n",
        "fig.set_figheight(5)\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1080x360 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 422
        },
        "id": "z1fitXsi93LE",
        "outputId": "2d034077-eb4e-43c8-a20b-b7b49730720a"
      },
      "source": [
        "fig,ax=plt.subplots(1,2)\n",
        "plt.xticks(rotation=45)\n",
        "sns.barplot(x='owner',y='selling_price',data=data2,ax=ax[0])\n",
        "sns.countplot(x='owner',data=data2,ax=ax[1])\n",
        "plt.xticks(rotation=45)\n",
        "fig.set_figwidth(15)\n",
        "fig.set_figheight(5)\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1080x360 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 352
        },
        "id": "yZOYPciv97Az",
        "outputId": "9999557a-9bd2-429c-973e-3d1b38265349"
      },
      "source": [
        "fig,ax=plt.subplots(1,2)\n",
        "plt.xticks(rotation=45)\n",
        "sns.barplot(x='years_old',y='selling_price',data=data2,ax=ax[0])\n",
        "sns.countplot(x='years_old',data=data2,ax=ax[1])\n",
        "plt.xticks(rotation=45)\n",
        "fig.set_figwidth(18)\n",
        "fig.set_figheight(5)\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1296x360 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_1SwP3rGFg0E"
      },
      "source": [
        "#### **Maping of Categorical Variables**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8RPd3J9pFIlP"
      },
      "source": [
        "mydict1={'Manual':1,'Automatic':2}\n",
        "mydict2={'Diesel':4,'Petrol':3,'LPG':2,'CNG':1}\n",
        "mydict3={'First Owner':4, 'Second Owner':3, 'Third Owner':2,'Fourth & Above Owner':1, 'Test Drive Car':5}\n",
        "mydict4={'Individual':1, 'Dealer':3, 'Trustmark Dealer':2}"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Kjc4zzBlKnXp"
      },
      "source": [
        "data2['transmission']=data2['transmission'].map(mydict1)\n",
        "data2['fuel']=data2['fuel'].map(mydict2)\n",
        "data2['owner']=data2['owner'].map(mydict3)\n",
        "data2['seller_type']=data2['seller_type'].map(mydict4)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "id": "lSAHw1zgLNjG",
        "outputId": "de4fc6f9-0157-490b-fbdc-0531e78228b4"
      },
      "source": [
        "data2.head()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>selling_price</th>\n",
              "      <th>km_driven</th>\n",
              "      <th>fuel</th>\n",
              "      <th>seller_type</th>\n",
              "      <th>transmission</th>\n",
              "      <th>owner</th>\n",
              "      <th>mileage</th>\n",
              "      <th>engine</th>\n",
              "      <th>max_power</th>\n",
              "      <th>seats</th>\n",
              "      <th>torque</th>\n",
              "      <th>years_old</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>385000</td>\n",
              "      <td>120000</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>19.10</td>\n",
              "      <td>1197.0</td>\n",
              "      <td>85.8</td>\n",
              "      <td>5.0</td>\n",
              "      <td>114.0</td>\n",
              "      <td>8</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>150000</td>\n",
              "      <td>80000</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>19.70</td>\n",
              "      <td>796.0</td>\n",
              "      <td>46.3</td>\n",
              "      <td>5.0</td>\n",
              "      <td>62.0</td>\n",
              "      <td>14</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>484999</td>\n",
              "      <td>75000</td>\n",
              "      <td>4</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>19.87</td>\n",
              "      <td>1461.0</td>\n",
              "      <td>83.8</td>\n",
              "      <td>5.0</td>\n",
              "      <td>200.0</td>\n",
              "      <td>6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>400000</td>\n",
              "      <td>70000</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>19.10</td>\n",
              "      <td>1197.0</td>\n",
              "      <td>85.8</td>\n",
              "      <td>5.0</td>\n",
              "      <td>114.0</td>\n",
              "      <td>7</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>70000</td>\n",
              "      <td>20000</td>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>15.90</td>\n",
              "      <td>1527.0</td>\n",
              "      <td>57.0</td>\n",
              "      <td>5.0</td>\n",
              "      <td>96.0</td>\n",
              "      <td>18</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   selling_price  km_driven  fuel  ...  seats  torque  years_old\n",
              "0         385000     120000     3  ...    5.0   114.0          8\n",
              "1         150000      80000     3  ...    5.0    62.0         14\n",
              "2         484999      75000     4  ...    5.0   200.0          6\n",
              "3         400000      70000     3  ...    5.0   114.0          7\n",
              "4          70000      20000     4  ...    5.0    96.0         18\n",
              "\n",
              "[5 rows x 12 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 207
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ro62XS27LfB_"
      },
      "source": [
        "#### **Multicollinearity Check**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Uc9cSV-BL3q2"
      },
      "source": [
        "from statsmodels.stats.outliers_influence import variance_inflation_factor"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 421
        },
        "id": "A4PVmOyoL4N4",
        "outputId": "ea4f9dc2-d39c-483e-e774-b67bb396bd9a"
      },
      "source": [
        "# calculating VIF for each feature\n",
        "vif_data = pd.DataFrame()\n",
        "vif_data[\"data2\"] = data2.columns\n",
        "vif_data[\"VIF_1\"] = [variance_inflation_factor(data2.values, i) for i in range(len(data2.columns))]\n",
        "vif_data.sort_values('VIF_1',ascending=False,ignore_index=True)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>data2</th>\n",
              "      <th>VIF_1</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>fuel</td>\n",
              "      <td>95.104680</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>seats</td>\n",
              "      <td>51.823991</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>engine</td>\n",
              "      <td>47.339003</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>mileage</td>\n",
              "      <td>39.911873</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>max_power</td>\n",
              "      <td>37.134368</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>owner</td>\n",
              "      <td>30.467614</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>transmission</td>\n",
              "      <td>19.277710</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>torque</td>\n",
              "      <td>14.114908</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>years_old</td>\n",
              "      <td>6.858649</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>km_driven</td>\n",
              "      <td>6.167965</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>10</th>\n",
              "      <td>seller_type</td>\n",
              "      <td>5.706243</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>11</th>\n",
              "      <td>selling_price</td>\n",
              "      <td>4.988098</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "            data2      VIF_1\n",
              "0            fuel  95.104680\n",
              "1           seats  51.823991\n",
              "2          engine  47.339003\n",
              "3         mileage  39.911873\n",
              "4       max_power  37.134368\n",
              "5           owner  30.467614\n",
              "6    transmission  19.277710\n",
              "7          torque  14.114908\n",
              "8       years_old   6.858649\n",
              "9       km_driven   6.167965\n",
              "10    seller_type   5.706243\n",
              "11  selling_price   4.988098"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 209
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6LGlFjYwMS3Q"
      },
      "source": [
        "Multicollinearity level in most of the features are very high. Let's scale these features and try VIF once more."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "m2UNEnw0MAwn"
      },
      "source": [
        "# Scaling all the features except Selling_price\n",
        "scaled=StandardScaler().fit_transform(data2[['km_driven','mileage','engine','max_power','seats','torque','years_old','fuel','seller_type','transmission','owner']])"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1XNE1rgINheV"
      },
      "source": [
        "data2_scaled=data2.copy()\n",
        "data2_scaled[['km_driven','mileage','engine','max_power','seats','torque','years_old','fuel','seller_type','transmission','owner']]=scaled"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "id": "AsMChprBNjQp",
        "outputId": "999620d3-7c33-4776-cd60-a61a36df45b6"
      },
      "source": [
        "data2_scaled.head()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>selling_price</th>\n",
              "      <th>km_driven</th>\n",
              "      <th>fuel</th>\n",
              "      <th>seller_type</th>\n",
              "      <th>transmission</th>\n",
              "      <th>owner</th>\n",
              "      <th>mileage</th>\n",
              "      <th>engine</th>\n",
              "      <th>max_power</th>\n",
              "      <th>seats</th>\n",
              "      <th>torque</th>\n",
              "      <th>years_old</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>385000</td>\n",
              "      <td>1.186431</td>\n",
              "      <td>-0.953217</td>\n",
              "      <td>-0.437525</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>0.640081</td>\n",
              "      <td>-0.074804</td>\n",
              "      <td>-0.514652</td>\n",
              "      <td>-0.152674</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>-0.546375</td>\n",
              "      <td>0.198816</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>150000</td>\n",
              "      <td>0.268294</td>\n",
              "      <td>-0.953217</td>\n",
              "      <td>-0.437525</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>-2.148011</td>\n",
              "      <td>0.078209</td>\n",
              "      <td>-1.314519</td>\n",
              "      <td>-1.262564</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>-1.083659</td>\n",
              "      <td>1.682495</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>484999</td>\n",
              "      <td>0.153527</td>\n",
              "      <td>0.869782</td>\n",
              "      <td>2.421162</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>0.640081</td>\n",
              "      <td>0.121563</td>\n",
              "      <td>0.011943</td>\n",
              "      <td>-0.208871</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>0.342209</td>\n",
              "      <td>-0.295744</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>400000</td>\n",
              "      <td>0.038760</td>\n",
              "      <td>-0.953217</td>\n",
              "      <td>-0.437525</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>-0.753965</td>\n",
              "      <td>-0.074804</td>\n",
              "      <td>-0.514652</td>\n",
              "      <td>-0.152674</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>-0.546375</td>\n",
              "      <td>-0.048464</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>70000</td>\n",
              "      <td>-1.108912</td>\n",
              "      <td>0.869782</td>\n",
              "      <td>-0.437525</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>0.640081</td>\n",
              "      <td>-0.890877</td>\n",
              "      <td>0.143592</td>\n",
              "      <td>-0.961910</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>-0.732358</td>\n",
              "      <td>2.671615</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   selling_price  km_driven      fuel  ...    seats    torque  years_old\n",
              "0         385000   1.186431 -0.953217  ... -0.43237 -0.546375   0.198816\n",
              "1         150000   0.268294 -0.953217  ... -0.43237 -1.083659   1.682495\n",
              "2         484999   0.153527  0.869782  ... -0.43237  0.342209  -0.295744\n",
              "3         400000   0.038760 -0.953217  ... -0.43237 -0.546375  -0.048464\n",
              "4          70000  -1.108912  0.869782  ... -0.43237 -0.732358   2.671615\n",
              "\n",
              "[5 rows x 12 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 212
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 421
        },
        "id": "2uZeMkcxN_T3",
        "outputId": "256a25b2-32e2-4f4f-938a-c0879d1daea7"
      },
      "source": [
        "vif_data[\"data2_scaled\"] = data2_scaled.columns\n",
        "vif_data[\"VIF_2\"] = [variance_inflation_factor(data2_scaled.values, i) for i in range(len(data2_scaled.columns))]\n",
        "vif_data.sort_values('VIF_2',ascending=False,ignore_index=True)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>data2</th>\n",
              "      <th>VIF_1</th>\n",
              "      <th>data2_scaled</th>\n",
              "      <th>VIF_2</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>max_power</td>\n",
              "      <td>37.134368</td>\n",
              "      <td>max_power</td>\n",
              "      <td>5.163375</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>engine</td>\n",
              "      <td>47.339003</td>\n",
              "      <td>engine</td>\n",
              "      <td>5.023781</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>torque</td>\n",
              "      <td>14.114908</td>\n",
              "      <td>torque</td>\n",
              "      <td>3.941524</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>mileage</td>\n",
              "      <td>39.911873</td>\n",
              "      <td>mileage</td>\n",
              "      <td>2.674529</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>fuel</td>\n",
              "      <td>95.104680</td>\n",
              "      <td>fuel</td>\n",
              "      <td>2.357786</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>seats</td>\n",
              "      <td>51.823991</td>\n",
              "      <td>seats</td>\n",
              "      <td>2.299420</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>years_old</td>\n",
              "      <td>6.858649</td>\n",
              "      <td>years_old</td>\n",
              "      <td>2.294974</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>km_driven</td>\n",
              "      <td>6.167965</td>\n",
              "      <td>km_driven</td>\n",
              "      <td>1.778124</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>selling_price</td>\n",
              "      <td>4.988098</td>\n",
              "      <td>selling_price</td>\n",
              "      <td>1.720891</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>transmission</td>\n",
              "      <td>19.277710</td>\n",
              "      <td>transmission</td>\n",
              "      <td>1.692594</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>10</th>\n",
              "      <td>owner</td>\n",
              "      <td>30.467614</td>\n",
              "      <td>owner</td>\n",
              "      <td>1.398249</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>11</th>\n",
              "      <td>seller_type</td>\n",
              "      <td>5.706243</td>\n",
              "      <td>seller_type</td>\n",
              "      <td>1.254821</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "            data2      VIF_1   data2_scaled     VIF_2\n",
              "0       max_power  37.134368      max_power  5.163375\n",
              "1          engine  47.339003         engine  5.023781\n",
              "2          torque  14.114908         torque  3.941524\n",
              "3         mileage  39.911873        mileage  2.674529\n",
              "4            fuel  95.104680           fuel  2.357786\n",
              "5           seats  51.823991          seats  2.299420\n",
              "6       years_old   6.858649      years_old  2.294974\n",
              "7       km_driven   6.167965      km_driven  1.778124\n",
              "8   selling_price   4.988098  selling_price  1.720891\n",
              "9    transmission  19.277710   transmission  1.692594\n",
              "10          owner  30.467614          owner  1.398249\n",
              "11    seller_type   5.706243    seller_type  1.254821"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 213
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 294
        },
        "id": "-vL0XNMFUJwG",
        "outputId": "662537e1-11e2-4bbe-9d9e-a284b600aa6d"
      },
      "source": [
        "fig, axes = plt.subplots(1,2,figsize=(15,4))\n",
        "axes[0].set_title('selling_price')\n",
        "sns.distplot(data2['selling_price'].values.reshape(-1,1),ax=axes[0],kde_kws = {'shade': True, 'linewidth': 3,'color':'black','fill':True},color='blue')\n",
        "axes[1].set_title('log-transformation')\n",
        "sns.distplot(np.log(data2['selling_price'].values.reshape(-1,1)),ax=axes[1],kde_kws = {'shade': True, 'linewidth': 3,'color':'black','fill':True},color='blue')\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1080x288 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "DNL1ilddV3cT"
      },
      "source": [
        "data2_scaled['selling_price']=np.log(data2_scaled['selling_price'])"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "id": "YtsvyciyU4Sm",
        "outputId": "0847e553-3437-4221-db03-7683db2b69c1"
      },
      "source": [
        "data2_scaled.head()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>selling_price</th>\n",
              "      <th>km_driven</th>\n",
              "      <th>fuel</th>\n",
              "      <th>seller_type</th>\n",
              "      <th>transmission</th>\n",
              "      <th>owner</th>\n",
              "      <th>mileage</th>\n",
              "      <th>engine</th>\n",
              "      <th>max_power</th>\n",
              "      <th>seats</th>\n",
              "      <th>torque</th>\n",
              "      <th>years_old</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>12.860999</td>\n",
              "      <td>1.186431</td>\n",
              "      <td>-0.953217</td>\n",
              "      <td>-0.437525</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>0.640081</td>\n",
              "      <td>-0.074804</td>\n",
              "      <td>-0.514652</td>\n",
              "      <td>-0.152674</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>-0.546375</td>\n",
              "      <td>0.198816</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>11.918391</td>\n",
              "      <td>0.268294</td>\n",
              "      <td>-0.953217</td>\n",
              "      <td>-0.437525</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>-2.148011</td>\n",
              "      <td>0.078209</td>\n",
              "      <td>-1.314519</td>\n",
              "      <td>-1.262564</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>-1.083659</td>\n",
              "      <td>1.682495</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>13.091902</td>\n",
              "      <td>0.153527</td>\n",
              "      <td>0.869782</td>\n",
              "      <td>2.421162</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>0.640081</td>\n",
              "      <td>0.121563</td>\n",
              "      <td>0.011943</td>\n",
              "      <td>-0.208871</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>0.342209</td>\n",
              "      <td>-0.295744</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>12.899220</td>\n",
              "      <td>0.038760</td>\n",
              "      <td>-0.953217</td>\n",
              "      <td>-0.437525</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>-0.753965</td>\n",
              "      <td>-0.074804</td>\n",
              "      <td>-0.514652</td>\n",
              "      <td>-0.152674</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>-0.546375</td>\n",
              "      <td>-0.048464</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>11.156251</td>\n",
              "      <td>-1.108912</td>\n",
              "      <td>0.869782</td>\n",
              "      <td>-0.437525</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>0.640081</td>\n",
              "      <td>-0.890877</td>\n",
              "      <td>0.143592</td>\n",
              "      <td>-0.961910</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>-0.732358</td>\n",
              "      <td>2.671615</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   selling_price  km_driven      fuel  ...    seats    torque  years_old\n",
              "0      12.860999   1.186431 -0.953217  ... -0.43237 -0.546375   0.198816\n",
              "1      11.918391   0.268294 -0.953217  ... -0.43237 -1.083659   1.682495\n",
              "2      13.091902   0.153527  0.869782  ... -0.43237  0.342209  -0.295744\n",
              "3      12.899220   0.038760 -0.953217  ... -0.43237 -0.546375  -0.048464\n",
              "4      11.156251  -1.108912  0.869782  ... -0.43237 -0.732358   2.671615\n",
              "\n",
              "[5 rows x 12 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 216
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ODU_uyijWDRz"
      },
      "source": [
        "X=data2_scaled.drop('selling_price',axis=1)\n",
        "y=data2_scaled['selling_price']"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "q99gmSj4Weuw",
        "outputId": "190714e8-b376-4b56-8743-682e06b0e657"
      },
      "source": [
        "# OLS\n",
        "import statsmodels.api as sm\n",
        "X= sm.add_constant(X)\n",
        "result = sm.OLS(y,X).fit()\n",
        "# printing the summary table\n",
        "print(result.summary())"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "                            OLS Regression Results                            \n",
            "==============================================================================\n",
            "Dep. Variable:          selling_price   R-squared:                       0.872\n",
            "Model:                            OLS   Adj. R-squared:                  0.872\n",
            "Method:                 Least Squares   F-statistic:                     5030.\n",
            "Date:                Tue, 20 Jul 2021   Prob (F-statistic):               0.00\n",
            "Time:                        15:51:45   Log-Likelihood:                -1749.9\n",
            "No. Observations:                8128   AIC:                             3524.\n",
            "Df Residuals:                    8116   BIC:                             3608.\n",
            "Df Model:                          11                                         \n",
            "Covariance Type:            nonrobust                                         \n",
            "================================================================================\n",
            "                   coef    std err          t      P>|t|      [0.025      0.975]\n",
            "--------------------------------------------------------------------------------\n",
            "const           12.9734      0.003   3894.601      0.000      12.967      12.980\n",
            "km_driven       -0.0492      0.004    -11.117      0.000      -0.058      -0.041\n",
            "fuel             0.0654      0.005     12.789      0.000       0.055       0.075\n",
            "seller_type      0.0410      0.004     11.026      0.000       0.034       0.048\n",
            "transmission     0.0628      0.004     14.643      0.000       0.054       0.071\n",
            "owner            0.0355      0.004      9.011      0.000       0.028       0.043\n",
            "mileage          0.0647      0.005     11.883      0.000       0.054       0.075\n",
            "engine           0.1329      0.007     17.816      0.000       0.118       0.148\n",
            "max_power        0.3566      0.007     48.475      0.000       0.342       0.371\n",
            "seats            0.0346      0.005      6.848      0.000       0.025       0.044\n",
            "torque           0.0109      0.007      1.644      0.100      -0.002       0.024\n",
            "years_old       -0.4116      0.005    -81.784      0.000      -0.421      -0.402\n",
            "==============================================================================\n",
            "Omnibus:                      557.438   Durbin-Watson:                   2.010\n",
            "Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2114.322\n",
            "Skew:                          -0.257   Prob(JB):                         0.00\n",
            "Kurtosis:                       5.445   Cond. No.                         5.67\n",
            "==============================================================================\n",
            "\n",
            "Warnings:\n",
            "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DmOw12lhk9Po"
      },
      "source": [
        "#### **Modelling**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "EerMHA1uWoC2"
      },
      "source": [
        "X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,shuffle=True,random_state=12)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zLO_8qQ-f1AS"
      },
      "source": [
        "# LINEAR REGRESSION\n",
        "def linear_regression(X_train,y_train,X_test,y_test):\n",
        "  y_pred=LinearRegression().fit(X,y).predict(X_test)\n",
        "  return (mean_squared_error(y_test,y_pred))\n",
        "\n",
        "#RIDGE\n",
        "def ridge_regression(X_train,y_train,X_test,y_test):\n",
        "  y_pred=Ridge(alpha=0.5).fit(X_train,y_train).predict(X_test)\n",
        "  return (mean_squared_error(y_test,y_pred))\n",
        "\n",
        "#LASSO\n",
        "def lasso_regression(X_train,y_train,X_test,y_test):\n",
        "  y_pred=Lasso(alpha=0.5).fit(X_train,y_train).predict(X_test)\n",
        "  return (mean_squared_error(y_test,y_pred))\n",
        "\n",
        "#KNN\n",
        "def knn_regression(X_train,y_train,X_test,y_test):\n",
        "  y_pred=KNeighborsRegressor(n_neighbors=2).fit(X_train,y_train).predict(X_test)\n",
        "  return (mean_squared_error(y_test,y_pred))\n",
        "\n",
        "#RANDOM FOREST\n",
        "def randomforest_reg(X_train,y_train,X_test,y_test):\n",
        "  y_pred=RandomForestRegressor(max_depth=3).fit(X_train,y_train).predict(X_test)\n",
        "  return (mean_squared_error(y_test,y_pred))\n",
        "#BAGGING\n",
        "def bagging_reg(X_train,y_train,X_test,y_test):\n",
        "  y_pred=BaggingRegressor(base_estimator=SVR(),n_estimators=10, random_state=0).fit(X_train,y_train).predict(X_test)\n",
        "  return (mean_squared_error(y_test,y_pred))\n",
        "\n",
        "#ADABOOST\n",
        "def adaboost_reg(X_train,y_train,X_test,y_test):\n",
        "  y_pred=AdaBoostRegressor().fit(X_train,y_train).predict(X_test)\n",
        "  return (mean_squared_error(y_test,y_pred))\n",
        "\n",
        "#XGBOOST\n",
        "def xgboost_reg(X_train,y_train,X_test,y_test):\n",
        "  y_pred=xgb.XGBRFRegressor(objective='reg:linear',n_estimators=10).fit(X_train,y_train).predict(X_test)\n",
        "  return (mean_absolute_error(y_test,y_pred))\n",
        "\n",
        "#SGDRegressor\n",
        "def sgd_reg(X_train,y_train,X_test,y_test):\n",
        "  y_pred=SGDRegressor().fit(X_train,y_train).predict(X_test)\n",
        "  return (mean_squared_error(y_test,y_pred))\n",
        "\n",
        "#EXTRA_TREES\n",
        "def extra_trees_reg(X_train,y_train,X_test,y_test):\n",
        "  y_pred=ExtraTreesRegressor().fit(X_train,y_train).predict(X_test)\n",
        "  return (mean_squared_error(y_test,y_pred))\n",
        "\n",
        "#GRADIENT BOOSTING\n",
        "def gradientB_reg(X_train,y_train,X_test,y_test):\n",
        "  y_pred=GradientBoostingRegressor().fit(X_train,y_train).predict(X_test)\n",
        "  return (mean_squared_error(y_test,y_pred))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "fdxDKviugCJq"
      },
      "source": [
        "def model(X,y,test_size):\n",
        "  #X=dataset.drop('selling_price',axis=1)\n",
        "  #y=dataset['selling_price']\n",
        "  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,shuffle=True,random_state=12)\n",
        "  #LINEAR REGRESSION\n",
        "  result=linear_regression(X_train,y_train,X_test,y_test)\n",
        "  #RIDGE\n",
        "  result1=ridge_regression(X_train,y_train,X_test,y_test)\n",
        "  #LASSO\n",
        "  result2=lasso_regression(X_train,y_train,X_test,y_test)\n",
        "  #KNN\n",
        "  result3=knn_regression(X_train,y_train,X_test,y_test)\n",
        "  #RANDOM FOREST\n",
        "  result4=randomforest_reg(X_train,y_train,X_test,y_test)\n",
        "  #BAGGING\n",
        "  result5=bagging_reg(X_train,y_train,X_test,y_test)\n",
        "  #ADABOOST\n",
        "  result6=adaboost_reg(X_train,y_train,X_test,y_test)\n",
        "  #XGBOOST\n",
        "  result7=xgboost_reg(X_train,y_train,X_test,y_test)\n",
        "  #SGDRegressor\n",
        "  result8=sgd_reg(X_train,y_train,X_test,y_test)\n",
        "  #EXTRA_TREES\n",
        "  result9=extra_trees_reg(X_train,y_train,X_test,y_test)\n",
        "  #GRADIENT BOOSTING\n",
        "  result10=gradientB_reg(X_train,y_train,X_test,y_test)\n",
        "  print('LINEAR: ',result,'\\n','RIDGE: ',result1,'\\n','LASSO:',result2,'\\n','KNN: ',result3,'\\n',\n",
        "        'RANDOMFOREST:',result4,'\\n','BAGGING:',result5,'\\n','ADABOOST:',result6,'\\n','XGBOOST:',result7,'\\n','SGDReg:',result8,'\\n',\n",
        "        'EXTRA_TREES:',result9,'\\n','GRADIENT_BOOSTING',result10)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WttFqCaSgFj9",
        "outputId": "20afb1ae-15d1-4d2a-caf5-fb71f4c2355f"
      },
      "source": [
        "model(X,y,0.3)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[15:52:00] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.\n",
            "LINEAR:  0.09036163183001017 \n",
            " RIDGE:  0.0907822389367786 \n",
            " LASSO: 0.5300514694802575 \n",
            " KNN:  0.07404908749842304 \n",
            " RANDOMFOREST: 0.14112004239538312 \n",
            " BAGGING: 0.05927558577840416 \n",
            " ADABOOST: 0.09765654956957463 \n",
            " XGBOOST: 0.31410153701792015 \n",
            " SGDReg: 0.09110138798011704 \n",
            " EXTRA_TREES: 0.04265919862696081 \n",
            " GRADIENT_BOOSTING 0.05304784739645403\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "aLZhofyyyQ8w"
      },
      "source": [
        "ExtraTreesRegressor and Gradient Boosting has least mean square error. Therefore adopting these algorithms for further analysis."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ppL2zYob0ggr"
      },
      "source": [
        "#performace metric\n",
        "def metric(y_pred,y_test):\n",
        "  y_pred1=np.exp(y_pred)\n",
        "  y_test1=np.exp(y_test)\n",
        "  print('MSE_log      =',round(mean_squared_error(y_pred,y_test),3))\n",
        "  print('RMSE_log     =',round(np.sqrt(mean_squared_error(y_pred,y_test)),3))\n",
        "  print('MAE_log      =',round(mean_absolute_error(y_pred,y_test),3))\n",
        "  print('r2_score     =',round(r2_score(y_pred,y_test),4))\n",
        "  print('MSE_exp      =',round(mean_squared_error(y_pred1,y_test1),3))\n",
        "  print('RMSE_exp     =',round(np.sqrt(mean_squared_error(y_pred1,y_test1)),3))\n",
        "  print('MAE_exp      =',round(mean_absolute_error(y_pred1,y_test1),3))\n",
        "  print('r2_score_exp =',round(r2_score(y_pred1,y_test1),3))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mp9oSanZoHJN",
        "outputId": "fb75e4cc-a205-4b66-90df-395df616eb54"
      },
      "source": [
        "X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,shuffle=True,random_state=12)\n",
        "y_pred=ExtraTreesRegressor().fit(X_train,y_train).predict(X_test)\n",
        "metric(y_pred,y_test)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "MSE_log      = 0.043\n",
            "RMSE_log     = 0.206\n",
            "MAE_log      = 0.137\n",
            "r2_score     = 0.9377\n",
            "MSE_exp      = 13391831948.89\n",
            "RMSE_exp     = 115723.083\n",
            "MAE_exp      = 60446.028\n",
            "r2_score_exp = 0.98\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QWHvQlf4mm9z",
        "outputId": "f08aa266-00ef-44d5-cb88-ebf10e07eba1"
      },
      "source": [
        "X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,shuffle=True,random_state=12)\n",
        "y_pred=ExtraTreesRegressor().fit(X_train,y_train).predict(X_test)\n",
        "metric(y_pred,y_test)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "MSE_log      = 0.042\n",
            "RMSE_log     = 0.206\n",
            "MAE_log      = 0.136\n",
            "r2_score     = 0.9381\n",
            "MSE_exp      = 13533681535.938\n",
            "RMSE_exp     = 116334.352\n",
            "MAE_exp      = 60467.272\n",
            "r2_score_exp = 0.98\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kdt3Ja08oWQX",
        "outputId": "09e747f1-2113-4ed3-8d8e-55592cc16210"
      },
      "source": [
        "y_pred=GradientBoostingRegressor().fit(X_train,y_train).predict(X_test)\n",
        "metric(y_pred,y_test)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "MSE_log      = 0.053\n",
            "RMSE_log     = 0.23\n",
            "MAE_log      = 0.169\n",
            "r2_score     = 0.9195\n",
            "MSE_exp      = 36200363376.726\n",
            "RMSE_exp     = 190263.931\n",
            "MAE_exp      = 94525.228\n",
            "r2_score_exp = 0.942\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OEkf1nEKsAWf",
        "outputId": "2e4f4cb5-1a7e-4bf1-be6c-b8c609f6d084"
      },
      "source": [
        "y_pred=GradientBoostingRegressor().fit(X_train,y_train).predict(X_test)\n",
        "metric(y_pred,y_test)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "MSE_log      = 0.053\n",
            "RMSE_log     = 0.23\n",
            "MAE_log      = 0.169\n",
            "r2_score     = 0.9195\n",
            "MSE_exp      = 36450504378.488\n",
            "RMSE_exp     = 190920.152\n",
            "MAE_exp      = 94577.452\n",
            "r2_score_exp = 0.942\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "PgaaJBfD55rb"
      },
      "source": [
        "ExtraTreesRegression is performing better than GradientBoostingRegressor"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 291
        },
        "id": "wLmQHs7tnnzX",
        "outputId": "a954d88c-7837-47b5-b166-84a0bbe55b5b"
      },
      "source": [
        "fig,ax=plt.subplots(1,2)\n",
        "sns.distplot((np.exp(y_test)-np.exp(y_pred)),ax=ax[0])\n",
        "plt.scatter(np.exp(y_test),np.exp(y_pred))\n",
        "fig.set_figwidth(15)\n",
        "fig.set_figheight(4)\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1080x288 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "E9sxVUErur0U",
        "outputId": "0a2d8b94-12bc-43a6-8369-ef09d9da0a94"
      },
      "source": [
        "per=np.percentile(data2['selling_price'].values,[5,25,50,75,85,95,99])\n",
        "per"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([ 110000.,  254999.,  450000.,  675000.,  850000., 1950000.,\n",
              "       5200000.])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 229
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "DM0oPfl66Yfv"
      },
      "source": [
        "def vals(x):\n",
        "  if x<per[0]:\n",
        "    return 5\n",
        "  elif per[0]<=x<per[1]:\n",
        "    return 25\n",
        "  elif per[1]<=x<per[2]:\n",
        "    return 50\n",
        "  elif per[2]<=x<per[3]:\n",
        "    return 75\n",
        "  elif per[3]<=x<per[4]:\n",
        "    return 85\n",
        "  elif per[4]<=x<per[5]:\n",
        "    return 95\n",
        "  else:\n",
        "    return 99"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3KKMHwrYCLHl"
      },
      "source": [
        "Adding a percentile feature that classify selling price and also give weightage."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YDA8d3yCBWny"
      },
      "source": [
        "data2_scaled['percentile']=data2['selling_price'].apply(lambda x:vals(x))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "id": "U7UTygPmBD58",
        "outputId": "1585b798-df9c-41be-9866-698ba2cf9dbe"
      },
      "source": [
        "data2_scaled.head()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>selling_price</th>\n",
              "      <th>km_driven</th>\n",
              "      <th>fuel</th>\n",
              "      <th>seller_type</th>\n",
              "      <th>transmission</th>\n",
              "      <th>owner</th>\n",
              "      <th>mileage</th>\n",
              "      <th>engine</th>\n",
              "      <th>max_power</th>\n",
              "      <th>seats</th>\n",
              "      <th>torque</th>\n",
              "      <th>years_old</th>\n",
              "      <th>percentile</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>12.860999</td>\n",
              "      <td>1.186431</td>\n",
              "      <td>-0.953217</td>\n",
              "      <td>-0.437525</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>0.640081</td>\n",
              "      <td>-0.074804</td>\n",
              "      <td>-0.514652</td>\n",
              "      <td>-0.152674</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>-0.546375</td>\n",
              "      <td>0.198816</td>\n",
              "      <td>50</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>11.918391</td>\n",
              "      <td>0.268294</td>\n",
              "      <td>-0.953217</td>\n",
              "      <td>-0.437525</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>-2.148011</td>\n",
              "      <td>0.078209</td>\n",
              "      <td>-1.314519</td>\n",
              "      <td>-1.262564</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>-1.083659</td>\n",
              "      <td>1.682495</td>\n",
              "      <td>25</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>13.091902</td>\n",
              "      <td>0.153527</td>\n",
              "      <td>0.869782</td>\n",
              "      <td>2.421162</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>0.640081</td>\n",
              "      <td>0.121563</td>\n",
              "      <td>0.011943</td>\n",
              "      <td>-0.208871</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>0.342209</td>\n",
              "      <td>-0.295744</td>\n",
              "      <td>75</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>12.899220</td>\n",
              "      <td>0.038760</td>\n",
              "      <td>-0.953217</td>\n",
              "      <td>-0.437525</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>-0.753965</td>\n",
              "      <td>-0.074804</td>\n",
              "      <td>-0.514652</td>\n",
              "      <td>-0.152674</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>-0.546375</td>\n",
              "      <td>-0.048464</td>\n",
              "      <td>50</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>11.156251</td>\n",
              "      <td>-1.108912</td>\n",
              "      <td>0.869782</td>\n",
              "      <td>-0.437525</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>0.640081</td>\n",
              "      <td>-0.890877</td>\n",
              "      <td>0.143592</td>\n",
              "      <td>-0.961910</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>-0.732358</td>\n",
              "      <td>2.671615</td>\n",
              "      <td>5</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   selling_price  km_driven      fuel  ...    torque  years_old  percentile\n",
              "0      12.860999   1.186431 -0.953217  ... -0.546375   0.198816          50\n",
              "1      11.918391   0.268294 -0.953217  ... -1.083659   1.682495          25\n",
              "2      13.091902   0.153527  0.869782  ...  0.342209  -0.295744          75\n",
              "3      12.899220   0.038760 -0.953217  ... -0.546375  -0.048464          50\n",
              "4      11.156251  -1.108912  0.869782  ... -0.732358   2.671615           5\n",
              "\n",
              "[5 rows x 13 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 232
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "STHABzwUCAJ_"
      },
      "source": [
        "X=data2_scaled.drop('selling_price',axis=1)\n",
        "y=data2_scaled['selling_price']\n",
        "X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,shuffle=True,random_state=12)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CxjwpDfaCf7L",
        "outputId": "f7172c38-c234-48c6-bfcc-93338c12104a"
      },
      "source": [
        "y_pred=ExtraTreesRegressor().fit(X_train,y_train).predict(X_test)\n",
        "metric(y_pred,y_test)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "MSE_log      = 0.018\n",
            "RMSE_log     = 0.135\n",
            "MAE_log      = 0.09\n",
            "r2_score     = 0.9745\n",
            "MSE_exp      = 8150411236.171\n",
            "RMSE_exp     = 90279.628\n",
            "MAE_exp      = 41515.517\n",
            "r2_score_exp = 0.988\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JBJcLVEPCr8I"
      },
      "source": [
        "scaled=StandardScaler().fit_transform(data2[['km_driven','mileage','engine','max_power','seats','torque','years_old','fuel','seller_type','transmission','owner']])\n",
        "data2_scaled_mm=data2.copy()\n",
        "data2_scaled_mm[['km_driven','mileage','engine','max_power','seats','torque','years_old','fuel','seller_type','transmission','owner']]=scaled"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "oyS6NBQvDsiR"
      },
      "source": [
        "data2_scaled_mm['selling_price']=np.log(data2_scaled_mm['selling_price'])\n",
        "data2_scaled_mm['percentile']=data2['selling_price'].apply(lambda x:vals(x))\n",
        "data2_scaled_mm['percentile']=StandardScaler().fit_transform(data2_scaled_mm['percentile'].values.reshape(-1,1))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "id": "sqS2QrOQEKj8",
        "outputId": "b90b602f-4e01-495d-a89a-dcb1a61e9d80"
      },
      "source": [
        "data2_scaled_mm.head()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>selling_price</th>\n",
              "      <th>km_driven</th>\n",
              "      <th>fuel</th>\n",
              "      <th>seller_type</th>\n",
              "      <th>transmission</th>\n",
              "      <th>owner</th>\n",
              "      <th>mileage</th>\n",
              "      <th>engine</th>\n",
              "      <th>max_power</th>\n",
              "      <th>seats</th>\n",
              "      <th>torque</th>\n",
              "      <th>years_old</th>\n",
              "      <th>percentile</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>12.860999</td>\n",
              "      <td>1.186431</td>\n",
              "      <td>-0.953217</td>\n",
              "      <td>-0.437525</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>0.640081</td>\n",
              "      <td>-0.074804</td>\n",
              "      <td>-0.514652</td>\n",
              "      <td>-0.152674</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>-0.546375</td>\n",
              "      <td>0.198816</td>\n",
              "      <td>-0.359176</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>11.918391</td>\n",
              "      <td>0.268294</td>\n",
              "      <td>-0.953217</td>\n",
              "      <td>-0.437525</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>-2.148011</td>\n",
              "      <td>0.078209</td>\n",
              "      <td>-1.314519</td>\n",
              "      <td>-1.262564</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>-1.083659</td>\n",
              "      <td>1.682495</td>\n",
              "      <td>-1.279514</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>13.091902</td>\n",
              "      <td>0.153527</td>\n",
              "      <td>0.869782</td>\n",
              "      <td>2.421162</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>0.640081</td>\n",
              "      <td>0.121563</td>\n",
              "      <td>0.011943</td>\n",
              "      <td>-0.208871</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>0.342209</td>\n",
              "      <td>-0.295744</td>\n",
              "      <td>0.561161</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>12.899220</td>\n",
              "      <td>0.038760</td>\n",
              "      <td>-0.953217</td>\n",
              "      <td>-0.437525</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>-0.753965</td>\n",
              "      <td>-0.074804</td>\n",
              "      <td>-0.514652</td>\n",
              "      <td>-0.152674</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>-0.546375</td>\n",
              "      <td>-0.048464</td>\n",
              "      <td>-0.359176</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>11.156251</td>\n",
              "      <td>-1.108912</td>\n",
              "      <td>0.869782</td>\n",
              "      <td>-0.437525</td>\n",
              "      <td>-0.385158</td>\n",
              "      <td>0.640081</td>\n",
              "      <td>-0.890877</td>\n",
              "      <td>0.143592</td>\n",
              "      <td>-0.961910</td>\n",
              "      <td>-0.43237</td>\n",
              "      <td>-0.732358</td>\n",
              "      <td>2.671615</td>\n",
              "      <td>-2.015784</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   selling_price  km_driven      fuel  ...    torque  years_old  percentile\n",
              "0      12.860999   1.186431 -0.953217  ... -0.546375   0.198816   -0.359176\n",
              "1      11.918391   0.268294 -0.953217  ... -1.083659   1.682495   -1.279514\n",
              "2      13.091902   0.153527  0.869782  ...  0.342209  -0.295744    0.561161\n",
              "3      12.899220   0.038760 -0.953217  ... -0.546375  -0.048464   -0.359176\n",
              "4      11.156251  -1.108912  0.869782  ... -0.732358   2.671615   -2.015784\n",
              "\n",
              "[5 rows x 13 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 237
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lmaL2h0lBs2i"
      },
      "source": [
        "#### **FINAL MODEL**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7Vafr4roEl4O",
        "outputId": "7988b53d-4814-4daf-d3d6-b3ba452d2ebe"
      },
      "source": [
        "X=data2_scaled_mm.drop('selling_price',axis=1)\n",
        "y=data2_scaled_mm['selling_price']\n",
        "X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,shuffle=True,random_state=12)\n",
        "y_pred=ExtraTreesRegressor().fit(X_train,y_train).predict(X_test)\n",
        "metric(y_pred,y_test)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "MSE_log      = 0.018\n",
            "RMSE_log     = 0.134\n",
            "MAE_log      = 0.09\n",
            "r2_score     = 0.9746\n",
            "MSE_exp      = 9035183902.002\n",
            "RMSE_exp     = 95053.584\n",
            "MAE_exp      = 41713.571\n",
            "r2_score_exp = 0.987\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5zHy0tzWRQ0p",
        "outputId": "1d035d79-04f5-42b2-edef-bde3ec33c222"
      },
      "source": [
        "model=ExtraTreesRegressor()\n",
        "model.fit(X_train,y_train)\n",
        "y_pred=model.predict(X_test)\n",
        "metric(y_pred,y_test)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "MSE_log      = 0.018\n",
            "RMSE_log     = 0.134\n",
            "MAE_log      = 0.09\n",
            "r2_score     = 0.9748\n",
            "MSE_exp      = 8157591629.976\n",
            "RMSE_exp     = 90319.387\n",
            "MAE_exp      = 41285.439\n",
            "r2_score_exp = 0.988\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ojdLPX8gwcYb",
        "outputId": "a85556ae-ea86-47bd-8fa5-e5703276683d"
      },
      "source": [
        "X=data2_scaled_mm.drop('selling_price',axis=1)\n",
        "y=data2_scaled_mm['selling_price']\n",
        "X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,shuffle=True,random_state=12)\n",
        "y_pred=ExtraTreesRegressor(n_estimators=60, n_jobs=4, min_samples_split=9,\n",
        "                            min_samples_leaf=10).fit(X_train,y_train).predict(X_test)\n",
        "metric(y_pred,y_test)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "MSE_log      = 0.018\n",
            "RMSE_log     = 0.136\n",
            "MAE_log      = 0.101\n",
            "r2_score     = 0.9737\n",
            "MSE_exp      = 16654039096.88\n",
            "RMSE_exp     = 129050.529\n",
            "MAE_exp      = 55741.373\n",
            "r2_score_exp = 0.974\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LxeTdO5gHY0r"
      },
      "source": [
        "from xgboost import XGBRegressor"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QSEvdL9FtsdM",
        "outputId": "313ef0d9-d7c6-4a04-ad88-52fb9acb11f2"
      },
      "source": [
        "xgb=XGBRegressor()\n",
        "xgb.fit(X_train,y_train)\n",
        "y_pred=xgb.predict(X_test)\n",
        "metric(y_pred,y_test)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[15:52:12] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.\n",
            "MSE_log      = 0.02\n",
            "RMSE_log     = 0.142\n",
            "MAE_log      = 0.111\n",
            "r2_score     = 0.9713\n",
            "MSE_exp      = 18168476823.344\n",
            "RMSE_exp     = 134790.492\n",
            "MAE_exp      = 65648.62\n",
            "r2_score_exp = 0.972\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "DPJdt8KWOSsX"
      },
      "source": [
        "temp=pd.DataFrame()\n",
        "temp['actual']=np.exp(y_test)\n",
        "temp['predict']=np.exp(y_pred)\n",
        "temp['diff']=round((np.exp(y_test)-np.exp(y_pred))*100/np.exp(y_test),2)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "PHMEf6GQOiWI"
      },
      "source": [
        "x=temp.head(20)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 309
        },
        "id": "T6kmiBW8OjeG",
        "outputId": "80072583-9133-4ee1-ea43-290585ebb1c5"
      },
      "source": [
        "x.plot(kind='bar',figsize=(20,4))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f6f8ccc5bd0>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 245
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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VdZbb721es3MmTOZOXPmGrd3JoGOOeaY/oYHrOWYQBHRAXwAuL3bpq2AR7o8X1qukyRJkiRJamtLlixh/PjxHHbYYUyYMIGDDz6YF154gY6ODr7yla8wdepUfvazn3H11Vez6667MnXqVD7xiU+wcuVKAK688krGjx/P1KlTueiii1a/79lnn81xxx0HwBNPPMFBBx3E5MmTmTx5MrfccgsnnngiDz74IFOmTOGEE07odzn6nASKiJHAvwN/npnPrssfi4ijI2JeRMxbtmzZuryFJEmSJEnSoPv1r3/NMcccw6JFi9h4441Xd9MaPXo0CxYsYK+99uLv/u7vuOaaa1iwYAHTp0/nlFNO4aWXXuKoo47i0ksvZf78+Tz++OM9vv/xxx/P7rvvzl133cWCBQuYOHEis2fPZtttt2XhwoV873vf63cZ+pQEiojhFAmgn2TmRT3s8iiwdZfnY8t1b5CZczJzemZOHzNmzLrEK0mSJEmSNOi23nprZsyYAcDhhx/OzTffDMAhhxwCwG233cZ9993HjBkzmDJlCueccw4PP/wwixcvZty4cWy33XZEBIcffniP73/ttdfyxS9+EYD11luPUaNa3O2NPowJFBEB/AhYlJmnrGG3ucBxEXE+sDOwwvGAJEmSJElSXRTpkTc/33DDDQHITPbee2/OO++8N+y3cOHCwQmwD/rSEmgG8MfARyJiYfnYLyK+EBFfKPe5HHgIeAD4F6A1IxZJkiRJkiQNAb/97W+59dZbAfjpT3/Khz70oTds32WXXfjFL37BAw88AMDzzz/P/fffz/jx41myZAkPPvggwJuSRJ323HNPfvCDYrL1V199lRUrVrDRRhvx3HPPtawMvSaBMvPmzIzM3CEzp5SPyzPzzMw8s9wnM/PYzNw2Mydl5ryWRShJkiRJklSx973vfXz/+99nwoQJPP3006u7bnUaM2YMZ599Noceeig77LADu+66K4sXL2bEiBHMmTOH/fffn6lTp7L55pv3+P6nnXYa1113HZMmTWLatGncd999jB49mhkzZvD+97+/JQNDRzGr++CbPn16zptnrkiSJElDU8eJl/W6z5IRn+79jVo09bAkNdWiRYuYMGFCpTEsWbKEAw44gHvuuafSOLrr6f8mIuZn5vSe9l+rKeIlSZIkSZLUnkwCSZIkSZIkvYWOjo4h1wpoXZgE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\n",
            "text/plain": [
              "<Figure size 1440x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 261
        },
        "id": "L37yMaV0O3rj",
        "outputId": "9ddc3e9f-cfd9-421b-8dd3-ad68934bc47d"
      },
      "source": [
        "x1=temp.sort_values(by='diff',ascending=False)\n",
        "x2=x1[['actual','predict']].head(50)\n",
        "x2.plot(kind='bar',figsize=(20,4))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f6f8ecce550>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 246
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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2PLly5kzZw6vfOUr2XHHHXnJS17CHXfcwezZs3n729/O/Pnz+fKXv8ypp57K7rvvzvz583npS1/K7bffDsApp5zCnDlzmD9/Pl/96lfvi3vcccdx+OGHA3DDDTfwohe9iLlz5zJ37lx+8pOfcOSRR3LllVcyb9483va2t7X6P0wkuTMNmA98IjP/BPgdcOToX4qIQyNicUQsXrlyZatCSZIkSZIkTZYrrriCww47jGXLlrHJJpvcd7vUFltswdKlS3nOc57De97zHk477TSWLl3KggULOProo7nzzjs55JBD+MY3vsGSJUu4/vrrx4x/xBFHsNdee3HhhReydOlSdtppJ4466ii23357LrjgAj74wQ+2Kv9EkjsrgBWZeU7z84mUZM9qMnNhZi7IzAWzZs1qVShJkiRJkqTJsu2227LHHnsA8KpXvYof/ehHALzsZS8D4Oyzz+ayyy5jjz32YN68eXz2s5/l6quv5vLLL2e77bbjiU98IhHBq171qjHjn3766fzt3/4tAOuvvz6bbtrt7WdrnHMnM6+PiGsjYofMvALYB7is01JI0iSYfeS3xnx9+VH7T3JJJEmSJE0lETHmzxttVGalyUz23XdfvvCFL6z2exdccMHkFHANJrpa1huBz0fERcA84H31iiRJkiRJkjR5rrnmGs466ywATjjhBPbcc8/V3n/605/Oj3/8Y37+858D8Lvf/Y6f/vSnzJkzh+XLl3PllVcC3C/5M2KfffbhE5/4BFAmZ7711lvZeOONue222zop/4SSO5l5QXPL1S6ZeVBm/qaTT5ckSZIkSVrHdthhBz7+8Y+z44478pvf/Oa+W6hGzJo1i+OOO45XvOIV7LLLLuy+++5cfvnlzJgxg4ULF7L//vszf/58ttxyyzHjf/SjH+WMM85g5513Ztddd+Wyyy5jiy22YI899uApT3lK6wmVIzNbBRjLggULcvHixZ3HfbDw1hBp3XDfkyRJkqaWZcuWseOOO67TMixfvpwXvOAFXHLJJeu0HKONtW0iYklmLhj9uxO9LUuSJEmSJElTkMkdSZIkSZL0kDV79uwpN2pnbZnckSRJkiRJ6jGTO5IkSZIkaZ2pMRdw363tNjG5I0mSJEmS1okZM2Zw0003meAZkJncdNNNzJgxY8J/M61ieSRJkiRJksa1zTbbsGLFClauXLmuizKlzJgx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\n",
            "text/plain": [
              "<Figure size 1440x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "FPw08_sF5lTh"
      },
      "source": [
        "import pickle\n",
        "file = open(\"file.pkl\", \"wb\") # opening a new file in write mode\n",
        "pickle.dump(model, file) # dumping created model into a pickle file"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fpJ4kbl_RDH4",
        "outputId": "c32100a1-f965-41d1-e1c8-504264406391"
      },
      "source": [
        "pickle.load(open('file.pkl','rb'))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse',\n",
              "                    max_depth=None, max_features='auto', max_leaf_nodes=None,\n",
              "                    max_samples=None, min_impurity_decrease=0.0,\n",
              "                    min_impurity_split=None, min_samples_leaf=1,\n",
              "                    min_samples_split=2, min_weight_fraction_leaf=0.0,\n",
              "                    n_estimators=100, n_jobs=None, oob_score=False,\n",
              "                    random_state=None, verbose=0, warm_start=False)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 251
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "I2uFYBniV2nv"
      },
      "source": [
        "import joblib"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_FWva7i5nlYX",
        "outputId": "196d08dc-9b37-490e-c117-5ecc1755418e"
      },
      "source": [
        "joblib.dump(model,'car_model.pkl')"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['car_model.pkl']"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 253
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oxYZA1tTnhOR"
      },
      "source": [
        "### **SUMMARY**:\n",
        "Finally, after testing various models the ExtraTreesRegressor and XGBRegressor, with little hypertuning, performed well.The R2 score of both the models comes out to be **0.97**.\n",
        "\n",
        "### **Future scope**:\n",
        "The dataset lacks **Actual price** column, presence of which could improve the performance and reliability of the model significantly.Since, we all know that the moment car comes out of a showroom, its price starts to depreciate.Perhaps, we could use this factor in determining the current worth of a car.\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hKn_Wi5Sn2Uq"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}