import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import sys
import numpy as np
import os
import json
from huggingface_hub import login
import matplotlib.pyplot as plt
import logging

from secret import API_KEY
from TDNK import TaskDoNKey

inst_set_1 = [
    ["James", "Move the plant next to the bookshelf.", ["the plant", "the bookshelf"], ["(Plant, in_corner)", "(Bookshelf, against_wall)"]],
    ["Michael", "Place the remote control on the sofa.", ["the remote control", "the sofa"], ["(Remote control, on_coffee_table)", "(Sofa, against_wall)"]],
    ["Robert", "Hang the blanket over the bookshelf.", ["the blanket", "the bookshelf"], ["(Blanket, draped_over_sofa)", "(Bookshelf, against_wall)"]],
    ["James", "Slide the coffee table closer to the sofa.", ["the coffee table", "the sofa"], ["(Coffee table, in_center_of_room)", "(Sofa, against_wall)"]],
    ["Michael", "Put the book on the coffee table.", ["the book", "the coffee table"], ["(Book, on_bookshelf)", "(Coffee table, in_center_of_room)"]],
    ["Michelle", "Turn the ceiling fan off.", ["the ceiling fan"], ["(Ceiling fan, on_ceiling)"]],
    ["Michelle", "Close the curtains over the window.", ["the curtains", "the window"], ["(Curtains, drawn_open)", "(Window, behind_sofa)"]],
    ["William", "Move the floor lamp near the plant.", ["the floor lamp", "the plant"], ["(Floor lamp, next_to_sofa)", "(Plant, in_corner)"]],
    ["Tom", "Place a coaster on top of the tv stand.", ["a coaster", "the tv stand"], ["(Coaster, on_coffee_table)", "(TV stand, against_wall)"]],
    ["Tom", "Take the painting down from next to the bookshelf and lean it against the sofa.", ["the painting", "the bookshelf", "the sofa"], ["(Painting, hung_on_wall)", "(Bookshelf, against_wall)", "(Sofa, against_wall)"]]
]

inst_set_2 = [
                ["Tom", "Bring a cup of coffee to my office.", ["a cup", "coffee", "my office"], ["(Tom's mug, on_table)", "(caffeinated, coffee_option)", "(Tom’s office, office_1)"]],
                ["Bob", "Bring a cup of coffee to my office.", ["a cup", "coffee", "my office"], ["(paper cup, in drawer)", "(caffeinated, coffee_option)", "(Bob’s office, office_2)"]],
                ["Tom", "Move the laptop from the lobby to the conference room.", ["the laptop", "the lobby", "the conference room"], ["(Laptop, in_lobby)", "(Lobby, room_2)", "(Conference Room, room_1)"]],
                ["Mary", "Throw the paper in the trash.", ["the paper", "the trash"], ["(Scrap paper, on_bob_desk)", "(Trach can, in_lobby)"]],
                ["Tom", "Bring the notes to Mary's office.", ["the notes", "Mary's office"], ["(Meeting Notes, in_conference_room)", "(Mary's office, office_3)"]]
            ]

inst_set_3 = [
    ["Tom", "Slice the tomato on the cutting board using the knife.",
     ["the tomato", "the cutting board", "the knife"],
     ["(Tomato, on_cutting_board)", "(Cutting board, on_counter)", "(Knife, on_cutting_board)"]],

    ["Sarah", "Toast a slice from the loaf of bread in the toaster.",
     ["the loaf of bread", "the toaster"],
     ["(Loaf of bread, in_cabinet)", "(Toaster, next_to_microwave)"]],

    ["Tom", "Move the apple from the table into the refrigerator.",
     ["the apple", "the table", "the refrigerator"],
     ["(Apple, on_table)", "(Table, in_center_of_room)", "(Refrigerator, against_wall)"]],

    ["Sarah", "Pour the bottle of milk into the glass in the dish rack.",
     ["the bottle of milk", "the glass", "the dish rack"],
     ["(Bottle of milk, in_refrigerator)", "(Glass, in_dish_rack)", "(Dish rack, next_to_sink)"]],

    ["Tom", "Take the egg carton out of the refrigerator and place it on the counter.",
     ["the egg carton", "the refrigerator", "the counter"],
     ["(Egg carton, in_refrigerator)", "(Refrigerator, against_wall)", "(Microwave, on_counter)"]],

    ["Sarah", "Move the pan from the stove to the egg carton.",
     ["the pan", "the stove", "the egg carton"],
     ["(Pan, on_stove)", "(Stove, part_of_oven)", "(Egg carton, in_refrigerator)"]],

    ["Tom", "Transfer the plate from the dish rack onto the table.",
     ["the plate", "the dish rack", "the table"],
     ["(Plate, in_dish_rack)", "(Dish rack, next_to_sink)", "(Table, in_center_of_room)"]],

    ["Sarah", "Put the banana into the trash can.",
     ["the banana", "the trash can"],
     ["(Banana, on_counter)", "(Trash can, near_door)"]],

    ["Tom", "Store the box of cereal from the top of refrigerator inside the cabinet.",
     ["the box of cereal", "refrigerator", "the cabinet"],
     ["(Box of cereal, on_top_of_refrigerator)", "(Refrigerator, against_wall)", "(Cabinet, above_counter)"]],

    ["Sarah", "Grate the cheese block and place the pieces on the plate on the table.",
     ["the cheese block", "the plate", "the table"],
     ["(Cheese block, in_refrigerator)", "(Plate, in_dish_rack)", "(Table, in_center_of_room)"]]
]

inst_set_4 = [
    ["Ana", "Attach the IV bag to the IV stand next to the hospital bed.",
     ["the IV bag", "the IV stand", "the hospital bed"],
     ["(IV bag, hanging_on_iv_stand)", "(IV stand, next_to_hospital_bed)", "(Hospital bed, in_room_101)"]],

    ["Dr. Singh", "Check the patient's temperature with the thermometer and record the result on the clipboard.",
     ["the patient", "the thermometer", "the clipboard"],
     ["(Patient, lying_on_hospital_bed)", "(Thermometer, in_nurse_pocket)", "(Clipboard, held_by_nurse)"]],

    ["James", "Move the wheelchair to the bedside table.",
     ["the wheelchair", "the bedside table"],
     ["(Wheelchair, in_corner_of_room_101)", "(Bedside table, next_to_hospital_bed)"]],

    ["Linda", "Turn off the heart monitor as directed by the doctor.",
     ["the heart monitor", "the doctor"],
     ["(Heart monitor, connected_to_patient)", "(Doctor, in_hallway)"]],

    ["Dr. Lee", "Administer the pill bottle to the patient.",
     ["the pill bottle", "the patient"],
     ["(Pill bottle, on_bedside_table)", "(Patient, lying_on_hospital_bed)"]],

    ["Ana", "Change the hospital gown on the patient to a fresh one.",
     ["the hospital gown", "the patient"],
     ["(Hospital gown, worn_by_patient)", "(Patient, lying_on_hospital_bed)"]],

    ["Taylor", "Place the X-ray image inside the medical folder and file it in the counter.",
     ["the X-ray image", "the medical folder", "the counter"],
     ["(X-ray image, in_medical_folder)", "(Medical folder, on_counter)", "(Counter, in_nurses_station)"]],

    ["Linda", "Apply the gloves to assist with checking the patient's vitals.",
     ["the gloves", "the patient"],
     ["(Gloves, worn_by_nurse)", "(Patient, lying_on_hospital_bed)"]],

    ["Jorge", "Refill the oxygen tank next to the hospital bed.",
     ["the oxygen tank", "the hospital bed"],
     ["(Oxygen tank, beside_bed)", "(Hospital bed, in_room_101)"]],

    ["Ana", "Remove the syringe and dispose of it in the trash can.",
     ["the syringe", "the trash can"],
     ["(Syringe, in_nurse_hand)", "(Trash can, outside_room)"]]
]

world_states = [('ws1.txt', inst_set_1), ('ws2.txt', inst_set_2), ('ws3.txt', inst_set_3), ('ws4.txt', inst_set_4)]

# world_states = [('ws2.txt', inst_set_2)]

system = TaskDoNKey()

for ws, inst_set in world_states:
    system.load_world_state(ws)

    stage1_acc = []
    stage2_acc = []
    stage3_acc = []

    cache = {}

    for speaker, inst, keyword_label, desc_label in tqdm(inst_set, desc="Running evaluation for current world state"):
        # Stage 1
        highlighted, keywords = system.highlight(inst)

        # Evaluate stage 1
        acc = len(set(keyword_label) & set(keywords)) / len(keyword_label)
        stage1_acc.append(acc)

        # Stage 2
        replacements = system.select_options(keyword_label, inst, speaker)

        # Evaluate stage 2
        s = 0
        for i, (k, v) in enumerate(replacements.items()):
            try:
                l = desc_label[i]
                if l in v:
                    s += 1
            except:
                pass
        s /= len(desc_label)
        stage2_acc.append(s)

        # Stage 3
        
        # Give options with definite solution
        for i, (k, v) in enumerate(replacements.items()):
            replacements[k] = list(set(v) | set([desc_label[i]]))
        
        selected, result = system.disamgibuate(replacements, cache, highlighted, speaker)

        # Evaluate stage 3
        s = 0
        for i, r in enumerate(selected):
            try:
                l = desc_label[i]
                if l == r:
                    s += 1
            except:
                pass
        s /= len(desc_label)
        stage3_acc.append(s)
    
    print(f"Stage 1 Accuracy: {sum(stage1_acc) / len(stage1_acc) * 100.0}")
    print(f"Stage 2 Accuracy: {sum(stage2_acc) / len(stage2_acc) * 100.0}")
    print(f"Stage 3 Accuracy: {sum(stage3_acc) / len(stage3_acc) * 100.0}")


# # instruct = input("Instruction: ")
# instruct = "Bring a cup of coffee to my office."

# # Stage 1
# highlighted, keywords = system.highlight(instruct)

# print(f"\nHighlighted non-specific keywords: {highlighted}\n")

# print(keywords)
# print()

# speaker = 'Tom'

# # Stage 2
# replacements = system.select_options(keywords, instruct, speaker)

# for k, v in replacements.items():
#     print(f"{k}:", replacements[k])
#     print()

# cache = {}

# result = system.disamgibuate(replacements, cache, highlighted, speaker)

# print(result)
