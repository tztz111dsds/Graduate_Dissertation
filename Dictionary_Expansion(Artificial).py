

# Code Summary:
# This section of code focuses on updating an existing dictionary of topics and keywords by adding new entries and
# saving the updated version to a CSV file. The process includes:
#
# 1. Reading the Existing Dictionary:
#    - The `load_dictionary_from_csv` function reads an existing CSV file and organizes it into a nested dictionary
#      format, with main topics as the primary keys, subtopics as secondary keys, and keywords as the values.
#
# 2. Defining New Keywords:
#    - The `new_keywords` dictionary contains new subtopics and keywords that need to be added to the existing dictionary.
#
# 3. Updating the Dictionary:
#    - The `update_dictionary` function adds the new keywords from `new_keywords` to the existing dictionary,
#      extending or adding to the relevant subtopics.
#
# 4. Removing Duplicates:
#    - The `remove_duplicates` function ensures that the updated dictionary does not contain duplicate keywords
#      within any subtopic.
#
# 5. Saving the Updated Dictionary:
#    - The `save_dictionary_to_csv` function saves the updated dictionary back into a CSV file.
#
# 6. Main Workflow:
#    - The `update_and_save_topics` function orchestrates the process by reading the existing dictionary, updating
#      it with new keywords, removing duplicates, and saving the final version.
#
# 7. Printing the Updated Dictionary:
#    - The `print_dictionary` function prints the content of the updated dictionary to the console for verification.

import csv

# 1. Function to read the existing dictionary from a CSV file
def load_dictionary_from_csv(file_name):
    topic_dict = {}
    with open(file_name, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            main_topic = row['Main Topic']
            sub_topic = row['Sub Topic']
            keyword = row['Keyword']
            if main_topic not in topic_dict:
                topic_dict[main_topic] = {}
            if sub_topic not in topic_dict[main_topic]:
                topic_dict[main_topic][sub_topic] = []
            topic_dict[main_topic][sub_topic].append(keyword)

    # Remove duplicates
    remove_duplicates(topic_dict)
    return topic_dict

# 2. New keywords dictionary (Modify new keywords here)
new_keywords = {
    'Property Layout & Amenities': {
        'General Layout': ['House'],
        'Space & Comfort': ['warm_inviting_ambiance', 'harmonious_blend_contemporary'],
    },
    'Location & Accessibility': {
        'Transport & Proximity': ['located_stone_throw'],
        'Nearby Landmarks & References': ['Covent_Garden_West_End', 'British_Museum_Tottenham_Court', 'neighborhood_St_Johns'],
        'Community & Neighborhood': ['scene_outside_neighbourhood']
    },
    'Design, Style & Quality': {
        'Style & Comfort': ['traditional_charm', 'tastefully_decorated'],
        'Luxury & Quality': ['interiors_showcase']
    },
    'Experience & Activities': {
        'User Experience': ['greeted_warm_inviting_ambiance', 'scene_outside'],
    }
}

# 3. Function to update the existing dictionary with new keywords
def update_dictionary(existing_dict, new_dict):
    for main_topic in new_dict:
        if main_topic in existing_dict:
            for sub_topic in new_dict[main_topic]:
                if sub_topic in existing_dict[main_topic]:
                    existing_dict[main_topic][sub_topic].extend(new_dict[main_topic][sub_topic])
                else:
                    existing_dict[main_topic][sub_topic] = new_dict[main_topic][sub_topic]
        else:
            existing_dict[main_topic] = new_dict[main_topic]

# 4. Function to remove duplicates from the dictionary
def remove_duplicates(topic_dict):
    for main_topic in topic_dict:
        for sub_topic in topic_dict[main_topic]:
            topic_dict[main_topic][sub_topic] = list(set(topic_dict[main_topic][sub_topic]))

# 5. Function to save the dictionary back into a CSV file
def save_dictionary_to_csv(topic_dict, file_name):
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Main Topic', 'Sub Topic', 'Keyword'])
        for main_topic, subtopics in topic_dict.items():
            for sub_topic, keywords in subtopics.items():
                for keyword in keywords:
                    writer.writerow([main_topic, sub_topic, keyword])

# 6. Main function to load, update, and save the dictionary
def update_and_save_topics(file_name, new_keywords):
    # Load the existing dictionary
    updated_topics = load_dictionary_from_csv(file_name)
    # Update the dictionary with new keywords
    update_dictionary(updated_topics, new_keywords)
    # Remove duplicates
    remove_duplicates(updated_topics)
    # Save the updated dictionary to a CSV file
    save_dictionary_to_csv(updated_topics, file_name)
    print(f"Dictionary has been saved to '{file_name}'")
    return updated_topics

# 7. Function to print the dictionary for verification
def print_dictionary(topic_dict):
    print("Updated dictionary content:")
    for category, subtopics in topic_dict.items():
        for subtopic, keywords in subtopics.items():
            print(f"{category} - {subtopic}: {keywords}")

# Example call
file_name = 'updated_topic_dictionary.csv'  # The dictionary file name you want to work with
updated_topics = update_and_save_topics(file_name, new_keywords)  # Update and save the dictionary
print_dictionary(updated_topics)  # Print the updated dictionary for verification
