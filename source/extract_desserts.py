import json
import numpy as np
from pattern.text.en import singularize

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
    non_dessert_identifier = ['soup','tacos', 'salad','casserole',
                    'pasta', 'meatloaf','fish','seafood','risotto'
                    ]
    return dessert_identifier, non_dessert_identifier

def not_desserts_vocabulary():
    """
    Get a loist of words that are not associated with a dessert. Like chicken, 
    so that chicken pot pir won't get classified as a dessert
    
    Output:
        List of words not being desserts.
    """
    not_dessert_vocab = ['fish','salmon','chicken','turkey','garlic']

    return not_dessert_vocab


def find_desserts(all_recipes, all_ingredients):
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
    for item,recipe in enumerate(all_recipes):
        #get the title and convert it into a list
        recipe_title = recipe['title'].lower().split()
        to_dessert_list = [word for word in recipe_title if word in dessert_ids]
        
        #if any word in the title is also a non dsert id, (soup, salad, etc...)
        # do not include it
        if any(singularize(word) in non_dessert_ids for word in recipe_title):
            continue

        if len(to_dessert_list)>0:
            dessert_list.append(recipe)
            ingredient_list.append(all_ingredients[item])
    
    return dessert_list, ingredient_list