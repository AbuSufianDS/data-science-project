"""
Complete Data Analysis Script
Saves all outputs as files - No GUI needed
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to non-interactive
plt.switch_backend('Agg')

print("🚀 Starting Data Analysis Pipeline")
print("=" * 50)

# 1. Generate sample data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'customer_id': range(1, n_samples + 1),
    'age': np.random.normal(35, 10, n_samples).clip(18, 70),
    'income': np.random.normal(50000, 15000, n_samples).clip(20000, 150000),
    'spending_score': np.random.normal(50, 20, n_samples).clip(1, 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
    'is_premium': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
})

# Add some relationships
data['income'] = data['income'] + data['age'] * 800
data['spending_score'] = data['spending_score'] + data['income'] / 5000

print("📊 Data Overview:")
print(f"   Samples: {len(data)}")
print(f"   Features: {len(data.columns)}")
print(f"   Columns: {', '.join(data.columns)}")

# 2. Save raw data
data.to_csv('data/raw_customer_data.csv', index=False)
print("💾 Saved raw data to: data/raw_customer_data.csv")

# 3. Generate summary statistics
summary = data.describe()
summary.to_csv('results/summary_statistics.csv')
print("📈 Generated summary statistics")

# 4. Create visualizations
print("🎨 Creating visualizations...")

# Figure 1: Distributions
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))

# Age distribution
axes1[0, 0].hist(data['age'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes1[0, 0].set_title('Age Distribution', fontsize=12)
axes1[0, 0].set_xlabel('Age')
axes1[0, 0].set_ylabel('Count')
axes1[0, 0].grid(True, alpha=0.3)

# Income distribution
axes1[0, 1].hist(data['income'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
axes1[0, 1].set_title('Income Distribution', fontsize=12)
axes1[0, 1].set_xlabel('Income ($)')
axes1[0, 1].set_ylabel('Count')
axes1[0, 1].grid(True, alpha=0.3)

# Spending score by region
sns.boxplot(data=data, x='region', y='spending_score', ax=axes1[1, 0])
axes1[1, 0].set_title('Spending Score by Region', fontsize=12)
axes1[1, 0].set_xlabel('Region')
axes1[1, 0].set_ylabel('Spending Score')
axes1[1, 0].grid(True, alpha=0.3)

# Income vs Spending Score
scatter = axes1[1, 1].scatter(data['age'], data['income'], 
                               c=data['spending_score'], 
                               alpha=0.6, cmap='viridis', s=20)
axes1[1, 1].set_title('Age vs Income (colored by Spending Score)', fontsize=12)
axes1[1, 1].set_xlabel('Age')
axes1[1, 1].set_ylabel('Income ($)')
plt.colorbar(scatter, ax=axes1[1, 1])

plt.tight_layout()
plt.savefig('results/distributions.png', dpi=150, bbox_inches='tight')
print("   Saved: results/distributions.png")

# Figure 2: Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_data = data.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig('results/correlation_matrix.png', dpi=150, bbox_inches='tight')
print("   Saved: results/correlation_matrix.png")

# 5. Statistical analysis
print("📊 Performing statistical analysis...")
premium = data[data['is_premium'] == 1]['spending_score']
non_premium = data[data['is_premium'] == 0]['spending_score']

t_stat, p_value = stats.ttest_ind(premium, non_premium)

with open('results/statistical_analysis.txt', 'w') as f:
    f.write("Statistical Analysis Results\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Premium customers: {len(premium)}\n")
    f.write(f"Non-premium customers: {len(non_premium)}\n\n")
    f.write(f"Mean spending (premium): ${premium.mean():.2f}\n")
    f.write(f"Mean spending (non-premium): ${non_premium.mean():.2f}\n\n")
    f.write(f"T-test results:\n")
    f.write(f"  t-statistic: {t_stat:.4f}\n")
    f.write(f"  p-value: {p_value:.6f}\n")
    f.write(f"  Significant at p < 0.05: {'YES' if p_value < 0.05 else 'NO'}\n")

print("   Saved: results/statistical_analysis.txt")

print("=" * 50)
print("✅ ANALYSIS COMPLETE!")
print("\n📁 Output files created:")
print("   data/raw_customer_data.csv")
print("   results/summary_statistics.csv")
print("   results/distributions.png")
print("   results/correlation_matrix.png")
print("   results/statistical_analysis.txt")
print("\n📋 To view results from Windows:")
print("   1. Open File Explorer")
print("   2. Type: \\\\wsl$\\Ubuntu\\home\\sufiands\\data_science_project")
print("   3. Open the 'results' folder")
