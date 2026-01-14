# Copilot instructions for this repository

Quick summary
- Small analysis scripts comparing classification algorithms on a 2-feature Iris dataset.
- Two main scripts: [iris_2feature2label_compare_res.py](iris_2feature2label_compare_res.py) (more complete) and [iris_2feature2label_compare.py](iris_2feature2label_compare.py) (earlier/partial).

What an AI agent should know (big picture)
- Purpose: load a small tabular dataset, train three classifiers (KNN, LogisticRegression, MLP), compute accuracy, and plot the 2D decision boundary.
- Data flow: CSV/XLSX -> pandas -> numpy arrays X (features) and y (labels) -> sklearn models -> predictions -> accuracy_score -> visualization via matplotlib.
- Structural choice: code assumes exactly two features for plotting. Many places slice X to X[:, :2] or expect 2-column inputs.

Key files & patterns (examples)
- Data-loading: iris_2feature2label_compare.py uses pd.read_csv and expects a Types label column. iris_2feature2label_compare_res.py uses pd.read_excel(header=0, index_col=0) and treats the last column as label.
- Visualization: draw_figure(X, y, clf) draws decision regions. It expects clf.predict semantics and 2D X. One script saves figures to figure.png; the other calls plt.show().
- Modeling: Algorithms used are KNeighborsClassifier, LogisticRegression(C=1e10), and MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,), random_state=1). Evaluation uses sklearn.metrics.accuracy_score.

Developer workflows & commands (what actually runs)
- Typical run (macOS / conda):
  ```bash
  conda activate base
  python iris_2feature2label_compare_res.py
  # or
  python iris_2feature2label_compare.py
  ```
- Dependencies to install if missing: numpy pandas matplotlib scikit-learn openpyxl
  ```bash
  pip install numpy pandas matplotlib scikit-learn openpyxl
  ```

Project-specific conventions and gotchas
- Data format: Excel loader expects index column at column 0 and label in last column. CSV loader expects a Types column for labels. Confirm which script you target before editing.
- Always ensure X passed to draw_figure is 2D with shape (n_samples, 2). Many functions slice to X[:, :2] — do not remove that unless you update plotting logic.
- Two variants in the repo: the _res file is the corrected/improved version. Prefer editing it for improvements, and mirror fixes back to the non-_res file only if intentional.

When making changes an AI should follow
- If changing data-loading, update both CSV and Excel callers and add clear comments about expected column names and shapes.
- When modifying a model signature or hyperparameters, keep prints of test accuracy (accuracy_score) so humans can quickly validate results.
- For visualization changes, preserve 2-feature requirement or update all places that assume it (X[:, :2], meshgrid generation, np.c_ usage).

Integration points & external dependencies
- No external services or APIs—everything is local. Files read by default: iris_2feature2label_train.csv / iris_2feature2label_test.csv or iris_2feature2label_train.xlsx / iris_2feature2label_test.xlsx.
- If adding CI or tests, assert that plotting code is non-blocking (use plt.close() in tests) and avoid plt.show() which blocks automated runs.

Notes for reviewers / next steps
- Would you like me to add a requirements.txt and a short README.md with run examples?
- Tell me if the project uses only CSV or prefers XLSX; I can normalize data-loading to one canonical format.

If anything above is unclear or you want different emphasis (e.g., focus on refactoring, tests, or containerization), tell me which area to expand.
