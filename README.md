# Leave room for dessert!

## Goal
The goals for this project are to analyze over 100k desset recipes found online to:
  - find unique flavor combinations
  - tell the used what type of dessert to make, given a certain number of ingredients
  - find an excuse to try new desserts

## Data
Certainly there are millions of dessert recipes out there, a simple Google search of 'dessert recipes' returns about 726 million results! Instead of spending the next few weeks (or months!) scrapping different recipe websites searching for the ultimate recipe, we use the data collected by [Marin et al.][1] for their project *Recipe1M+:  A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes and Food Images* published tin the IEEE Tansactions on Pattern Analysis and Machine Intelligence. 

The datasets, which can be accessed through [here](http://im2recipe.csail.mit.edu/), consist of over 1 million with 13 million associated images. For this project, I will only focus on the text recipes. The data is stored in two json files: 
  - layer1.json file, (1.3Gb) with 1,0290,720 recipes
  - det_ingrs.json file (344 Mb) with the extracted ingredients for each recipe in the layer1.json file.
  


### Finding desserts (data cleaning)
As with any data science project, the first and most time consuming step is cleaning and preparing the data so we can create a model with it. For this project the first step is extracting the desserts from the dataset. According to [Marin et al.][1], 22% of the data are desserts. 

## Running some model


## Summary

# Interesting finds...
Is ["Cornbread Turkay Taco Cheescake"](https://www.food.com/recipe/cornbread-turkey-taco-cheesecake-with-zesty-avocado-cream-398439), an entree? a dessert? Thanksgiving food? Mexican food? American for sure.

## Future work

[1]: https://ieeexplore.ieee.org/document/8758197

