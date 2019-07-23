import json
import numpy as np
from pattern.text.en import singularize
import re
import spacy
def get_raw_data(filename='sample_layer1.json'):
    """
    Open the json file, and load the data as a list
    
    Input:
        filename, default value contains 250 entries
    Output:
        list with all the recipes
    """
    data_dir = '../data/'
    file_path = data_dir + filename
    file_path = data_dir + filename
    print('Loading file ', file_path)
    with open(file_path,'r') as file:
        raw_data = json.load(file)

    return raw_data
def get_raw_ingredients(filename='sample_det_ingrs.json'):
    """
    Get the raw ingredients

    Input: 
        filename, default is a file containing 250 entries
    Output:
        list with the ingredients
    """
    data_dir = '../data/'
    file_path = data_dir + filename
    print('Loading ingredient file: ' + file_path)
    with open(file_path, 'r') as file:
        raw_ingredients = json.load(file)
    
    return raw_ingredients

def get_sweet_vocabulary():
    """
    Get a list of words associated with desserts
    Input:

    Output:
        List of words identifying desserts
    """
    dessert_identifier = ['cookies','pie','crumble', 'cake', 'ice-cream', 
                      'ice cream','praline','macaron','cheesecake','cupcake', 
                      'waffle','pudding','truffle','toffee','tart',
                      'torte', 'zabiglione','trifle','toffee','sweets',
                      'sundae','strudel','shortcake','souffle',
                      'shortbread','sherbet','scone'"s'mores",'bread',
                      'popsicle','popover','brittle','pastry','parfait',
                      'panna','cotta','nougat','muffin','mousse','meringue'
                      'marshmallow','macaroon','ladyfingers','tiramisu','jelly',
                      'icing','jellyroll','fudge', 'honey','gelato',
                      'gelatin','gingersnaps','gingerbread','frosting','fritter',
                      'flan','eclair','donut','doughnut','dessert','pastry',
                      'custard','crepe','cobbler','churro','caramel',
                      'cannoli','butterscotch','brownie','buttercream','bombe',
                      'biscotti','baklava','Alaska','cream','ice','crisp',
                      'chocolate','compote','confection','fondant','fro-yo',
                      'ganache','icing','jell-o','marzipan','molasses',
                      'melba','snow','sorbet','sweets','syrup','whip',
                      'jello','gateau','glacee','gelatine','tarte','bonbon',
                      'biscuits', 'blondie','buckeye','confection','candy',
                      'cupcone','hostess','oreo','snickerdoodles','twinkie',
                      'butter','parfaits','vanilla','leches', 'scone'
                     ]
    non_dessert_identifier = ['soup','taco', 'salad','casserole',
                    'pasta', 'meatloaf','fish','seafood','risotto','stew', 'savory','savoury',
                    'stir-fry','corn bread','steak','vinaigrette','pasta','ravioli', 'gnocchi'
                    ]
    return dessert_identifier, non_dessert_identifier

def not_dessert_ingredients():
    """
    Get a loist of words that are not associated with a dessert. Like chicken, 
    so that chicken pot pir won't get classified as a dessert
    
    Output:
        List of words not being desserts.
    """
    not_dessert_ingrs = ['fish','salmon','tuna','chicken','turkey','garlic', 'onion','lamb',
    'sausage','shrimp','beef', 'taco','shallot','veal','pork','mincemeat','crab','filet',
    'chipotle', 'panceta', 'asparagus','parsley','mushroom','sardines','olives','oyster','ham',
    'snow pea', 'kimchi','cilantro','Worcestershire','tomato paste', 'salsa','bologna']

    return not_dessert_ingrs



def clean_dessert_ingredients(all_ingredients):

    for item, ingredients_list in enumerate(all_ingredients):
        for ingr_item, ingredient in enumerate(ingredients_list['ingredients']):
            print(ingredient['text'])
        break
    return all_ingredients

def find_desserts(all_recipes, all_ingredients, test_id='000'):
    """
    Extract the dessert recipes from a list of recipes
    bsed on a list of identifier words
    Input:
        List of recipes and list of ingredients
    Output:
        List of dessert recipes, and a list of their ingredients
    """

    dessert_list = []
    ingredient_list = []
    dessert_ids, non_dessert_ids = get_sweet_vocabulary()
    non_dessert_ingredients = not_dessert_ingredients()
    # print(non_dessert_ingredients)
    for item,recipe in enumerate(all_recipes):
        #get the title and convert it into a list
        recipe_title = recipe['title'].lower().split()
        recipe_id = recipe['id']
        #if any word in the title is also a non dsert id, (soup, salad, etc...)
        # do not include it  
        if(test_id == recipe_id): print('1. \t test for non dessert words in title')     
        if any(singularize(word) in non_dessert_ids for word in recipe_title):
            continue

        # look at the ingredients, if any ingredient is in the not dessert ingredient list, do not add it.
        tmp_ingredient_list = []
        if(test_id == recipe_id): print('2. \t test for non dessert ingredients')
        found_savory_word =0
        for ingred_list in all_ingredients[item]['ingredients']:
            
            for bad_word in non_dessert_ingredients:
                if(test_id == recipe_id): print(bad_word,';  ',ingred_list['text'])
                if re.search(bad_word, ingred_list['text'].lower()):
                    # print(bad_word, ingred_list['text'])
                    found_savory_word = 1
                    if(test_id == recipe_id): input()
                    continue
        if found_savory_word==1:
            continue

        
        if(test_id == recipe_id): print('3. \t look for dessert words in title')
        # if any(singularize(word) in non_dessert_ingredients for word in tmp_ingredient_list):
            # continue

        to_dessert_list = [word for word in recipe_title if word in dessert_ids]
        

        
        if(test_id == recipe_id): print('4. \t if found a word in tite, add it')

        if len(to_dessert_list)>0:
            dessert_list.append(recipe)
            ingredient_list.append(all_ingredients[item])
    
    return dessert_list, ingredient_list