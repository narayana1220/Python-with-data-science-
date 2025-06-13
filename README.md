# 🎬 Netflix Movie Recommendation System

This project is a **collaborative filtering-based movie recommendation system** built using a subset of the Netflix Prize dataset. It leverages **Singular Value Decomposition (SVD)** from the `Surprise` library to recommend movies to users based on their past ratings.

---

## 📌 Project Goals

- Build a personalized movie recommendation system using collaborative filtering.
- Apply SVD to uncover hidden features between users and movies.
- Evaluate model performance using RMSE and Cross-Validation.
- Filter out noise by dropping low-rating-frequency users and movies.

---

## 🧠 Skills & Tools Used

- **Python**
- **Pandas, NumPy** – Data preprocessing
- **Matplotlib, Seaborn** – Visualization
- **Surprise** – Recommender system (SVD)
- **RMSE & Cross-validation** – Model evaluation

---

## 📁 Dataset Description

 Files Used:
1. **combined_data_1.txt.zip** – Netflix ratings data  
   - `Cust_Id`: User ID or Movie ID line (e.g., `"1:"`)
   - `Rating`: Movie rating (1 to 5) or NaN if row contains a Movie ID

2. **movie_titles.csv** – Metadata of movies  
   - `Movie_Id`: Unique ID of movie  
   - `Year`: Release year  
   - `Name`: Movie title  

---

## 🔧 Project Workflow

### 1. Data Preprocessing
- Extract `movie_Id` from lines containing `":"` and propagate to ratings.
- Remove NaN values (movie headers) to retain only rating rows.
- Convert `Cust_Id` to integers.

### 2. Data Filtering
- Remove movies rated by less than the top 60% of users.
- Remove users who rated fewer movies than the 60th percentile.

### 3. Model Training
```python
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

reader = Reader()
data = Dataset.load_from_df(filtered_df[['movie_Id', 'Cust_Id', 'Rating']][:100000], reader)
model = SVD()
cross_validate(model, data, measures=['RMSE'], cv=3)
