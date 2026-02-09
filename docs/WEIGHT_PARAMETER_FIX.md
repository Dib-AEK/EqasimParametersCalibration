# Weight Parameter Fix for Histogram Plotting

## Problem

The original code had an incorrect usage of the weight parameter in pandas' `.plot.hist()` method:

```python
# INCORRECT CODE
for purpose in mz_trips["purpose"].unique():
    mask_mz = (mz_trips["purpose"]==purpose)
    mask_matsim = (matsim_trips["purpose"]==purpose)
    
    bins = [0,1,2,3,5,7,10,15,20,35,500]    
    fig, ax = plt.subplots(figsize=(10,5))    
    mz_trips.loc[mask_mz, "euclidean_distance_km"].plot.hist(
        ax=ax, bins=bins, density=True, alpha=0.5, label="MZ", weight="person_weight"
    )    
    matsim_trips.loc[mask_matsim, "euclidean_distance_km"].plot.hist(
        ax=ax, bins=bins, density=True, alpha=0.5, label="MATSim"
    )    
    plt.xlim([0,50])
    plt.legend()
    plt.title(purpose)
    plt.show()
```

### Issues with the Original Code

1. **Wrong parameter name**: Used `weight` (singular) instead of `weights` (plural)
2. **Wrong parameter type**: Passed a string `"person_weight"` instead of the actual Series containing the weights
3. **Error**: This causes `AttributeError: Rectangle.set() got an unexpected keyword argument 'weight'`

## Solution

The corrected code passes the actual Series of weights using the correct parameter name:

```python
# CORRECTED CODE
for purpose in mz_trips["purpose"].unique():
    mask_mz = (mz_trips["purpose"]==purpose)
    mask_matsim = (matsim_trips["purpose"]==purpose)
    
    bins = [0,1,2,3,5,7,10,15,20,35,500]    
    fig, ax = plt.subplots(figsize=(10,5))    
    mz_trips.loc[mask_mz, "euclidean_distance_km"].plot.hist(
        ax=ax, bins=bins, density=True, alpha=0.5, label="MZ", 
        weights=mz_trips.loc[mask_mz, "person_weight"]  # FIXED: Pass actual Series
    )    
    matsim_trips.loc[mask_matsim, "euclidean_distance_km"].plot.hist(
        ax=ax, bins=bins, density=True, alpha=0.5, label="MATSim"
    )    
    plt.xlim([0,50])
    plt.xlabel("Euclidean Distance (km)")  # Added axis label
    plt.ylabel("Density")  # Added axis label
    plt.legend()
    plt.title(purpose)
    plt.show()
```

### Key Changes

1. **Parameter name**: Changed `weight=` to `weights=` (plural)
2. **Parameter value**: Changed `"person_weight"` to `mz_trips.loc[mask_mz, "person_weight"]`
3. **Added axis labels**: Improved plot readability

## Alternative Implementation

For better clarity and control, you can also use `matplotlib.pyplot.hist()` directly:

```python
for purpose in mz_trips["purpose"].unique():
    mask_mz = (mz_trips["purpose"]==purpose)
    mask_matsim = (matsim_trips["purpose"]==purpose)
    
    bins = [0,1,2,3,5,7,10,15,20,35,500]    
    fig, ax = plt.subplots(figsize=(10,5))
    
    # Extract data explicitly
    mz_distances = mz_trips.loc[mask_mz, "euclidean_distance_km"]
    mz_weights = mz_trips.loc[mask_mz, "person_weight"]
    matsim_distances = matsim_trips.loc[mask_matsim, "euclidean_distance_km"]
    
    # Plot with explicit weights
    ax.hist(mz_distances, bins=bins, density=True, alpha=0.5, 
            label="MZ", weights=mz_weights)
    ax.hist(matsim_distances, bins=bins, density=True, alpha=0.5, 
            label="MATSim")
    
    ax.set_xlim([0,50])
    ax.set_xlabel("Euclidean Distance (km)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(purpose)
    plt.show()
```

## Usage in Project

A utility function has been created at `utils/distance_comparison.py` that demonstrates the correct usage:

```python
from utils.distance_comparison import plot_distance_distributions_by_purpose

# Use with your MZ and MATSim trip data
plot_distance_distributions_by_purpose(mz_trips, matsim_trips)
```

## Technical Details

The `weights` parameter in both `pandas.Series.plot.hist()` and `matplotlib.pyplot.hist()`:
- Must be an array-like object (numpy array, pandas Series, list, etc.)
- Should have the same length as the data being plotted
- Each weight corresponds to one data point
- When `density=True`, the histogram is normalized to form a probability density

## Testing

A test script (`/tmp/test_weight_parameter.py`) validates:
- The incorrect usage fails with an appropriate error
- The correct usage succeeds and produces weighted histograms
- Alternative implementations work correctly
