# Leave room for dessert!

## Goal
The goals for this project are to analyze over 100k desset recipes found online to:
  - find unique flavor combinations
  - tell the used what type of dessert to make, given a certain number of ingredients
  - find an excuse to try new desserts

## Data
Certainly there are millions of dessert recipes out there, a simple Google search of 'dessert recipes' returns about 726 million results! Instead of spending the next few weeks (or months!) scrapping different recipe websites searching for the ultimate recipe, we use the data collected by [Marin et al.][1] for their project *Recipe1M+:  A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes and Food Images* published tin the IEEE Tansactions on Pattern Analysis and Machine Intelligence. 

The datasets, which can be accessed through [here](http://im2recipe.csail.mit.edu/), consist of over 1 million with 13 million associated images. For this project, I will only focus on the text recipes. The data is stored in two json files: 
  - layer1.json file, (1.3Gb) with 1,0290,720 recipes. 
  - det_ingrs.json file (344 Mb) with the extracted ingredients for each recipe in the layer1.json file.
  
 Each recipe int he layer1.json file is a dictionary with the following information: 

   | Key | Description |
   |:---:|:---:|
   |ingredients| A list of all the ingredients needed |
   | url | The url where the recipe was scrapped |
   | title | The title of the dish |
   | id | Unique id given to the recipe |
   | instructions | List of instructions on how to make the dish |
    
  Below is an example of one of the recipes
    
   | Key | Data |
   |:---:|:---:|
   |ingredients| 1 whole Vanilla Wafer Cookie, 1- 1/2 whole Peeled Bananas,'2 cups Coconut Milk, 2 Tablespoons Fat-free Half-and-half, 1 Tablespoon Vanilla Extract, 1/2 teaspoons Stevia (more To Taste), 1 cup Ice Cubes,4 Tablespoons Whipped Cream|
   | url | http://tastykitchen.com/recipes/drinks/banana-cream-pie-smoothie/ |
   |title | Banana Cream Pie Smoothie |
   | id | 00016355e6 |
   | instructions | Place vanilla wafer cookie in a baggie, seal bag and crush it into tiny pieces (I rolled a rolling pin over it). Set aside. In a blender add banana, coconut milk, half-and-half, vanilla extract, stevia and ice cubes. Put the lid on and blend until smooth. Pour into 2 cups and top each smoothie with whipped cream and crushed wafer cookie. |
   
The second file (det_ingrs.json) contains the ingredients extracted by [Marin et al.][1]. For each recipe, there is a dictionary with the keys `id`, and `ingredients`. The table below shows the values for the recipe  shown above:

  | Key | Data |
  |:---:|:---:|
  |id | 00016355e6 |
  |ingredients | vanilla wafer cookie, bananas, coconut milk, 'fat-free half-and-half, vanilla extract, stevia, ice cubes, whipped cream|

  
### Finding desserts (data cleaning)
As with any data science project, the first and most time consuming step is cleaning and preparing the data so we can create a model with it. For this project the first step is extracting the desserts from the dataset. According to [Marin et al.][1], 22% of the data are desserts. 

The first step is to identify desserts, for that I look for "dessert keywords" in the recipe title. Some of these "dessert keywords" are
```python
dessert_keywords =    ['cookies','pie','crumble', 'cake', 'ice-cream','ice cream','praline','macaron',
                      'cheesecake','cupcake','waffle','pudding','truffle','toffee','tart','torte',
                      'zabiglione','trifle','toffee','sweets',  'sundae','strudel','shortcake','souffle',
                      'shortbread','sherbet','scone'"s'mores",'bread','popsicle','popover','brittle', ... ]
```                     
Selecting desserts this way results in about 23% of the recipes classified as desserts. However, there are some outliers, for example: ["Cornbread Turkay Taco Cheescake"](https://www.food.com/recipe/cornbread-turkey-taco-cheesecake-with-zesty-avocado-cream-398439). Is this an entree? a dessert? Mexican food? 
To further clean the data, I rejected files that had a non-dessert keyword on the title:
```python
non_dessert_keywords = ['soup','taco','salad','casserole','pasta','meatloaf','fish','seafood',
                        'risotto','stew','savory','savoury','stir-fry','corn bread','steak','pasta', ... ]
```
However, if watching all the seasons of "The Great British Bake Off" has taught me anything, is that there are savory pies, and that cookies may be called biscuits. So, to further clean the data scanned through the ingredeints in the recipe, looking for non-dessert ingredients. This step is a bit tricky as I want to keep unusual combinations, such as "kale chocolate chip cookies" or "bacon donuts", however, I can't imagine any dessert containing any sort of fish in it. The final list of ingredients to reject is:
```python
not_dessert_ingrs = ['fish','salmon','tuna','chicken','turkey','garlic', 'onion','lamb',
    'sausage','shrimp','beef', 'taco','shallot','veal','pork','mincemeat','crab','filet',
    'chipotle', 'panceta', 'asparagus','parsley','mushroom','sardines','olives','oyster','ham',
    'snow pea', 'kimchi','cilantro','Worcestershire','tomato paste', 'salsa','bologna']
```
After filtering the data, I ended with 195,805 dessert recipes, this corresponds to ~19.1% of the data. 



![Histogram of number of ingredients](source/figs/Ingredients_Instructions_hist.png)


Finally to be able predict what type of dessert the recipe is trying to make, we split the the desserts into one of the following categories
```python
categories = ['cake', 'cookies', 'pie', 'pudding', 'other']
```
To be able to predict the catergoy, I reject the desserts labeled as `other`. 

## Running some model


## Summary

# Interesting finds...
, an entree? a dessert? Thanksgiving food? Mexican food? American for sure.

## Future work

![Alt Text](https://media.giphy.com/media/l3vRhl6k5tb3oPGLK/giphy.gif)

" width="480" height="270" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/season-6-2015-bbc-l3vRhl6k5tb3oPGLK">via GIPHY</a></p>


[1]: https://ieeexplore.ieee.org/document/8758197

