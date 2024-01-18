from typing import List

import torch


def merge_template_search(inp_list):
    if isinstance(inp_list[0]['feat'], List):
        lens = len(inp_list[0]['feat'])

        return {"feat": [torch.cat([x["feat"][i] for x in inp_list], dim=0) for i in range(lens)],
                "mask": [torch.cat([x["mask"][i] for x in inp_list], dim=1) for i in range(lens)],
                "pos": [torch.cat([x["pos"][i] for x in inp_list], dim=0) for i in range(lens)]}

    # return {"feat": torch.cat([x["feat"] for x in inp_list], dim=0),
    #         "mask": torch.cat([x["mask"] for x in inp_list], dim=1),
    #         "pos": torch.cat([x["pos"] for x in inp_list], dim=0)}

    return {"feat": torch.cat([x["feat"].clone() for x in inp_list], dim=0),
            "mask": torch.cat([x["mask"].clone() for x in inp_list], dim=1),
            "pos": torch.cat([x["pos"].clone() for x in inp_list], dim=0)}


def list_template_search(inp_list):
    return {"feat": [x["feat"] for x in inp_list],
            "mask": [x["mask"] for x in inp_list],
            "pos": [x["pos"] for x in inp_list]}


def merge_channel(inp_list):
    mask = torch.zeros_like(inp_list[0]["mask"])
    for x in inp_list:
        mask = mask | x["mask"]
    return {"feat": [x["feat"] for x in inp_list],
            "mask": mask,
            "pos": inp_list[0]["pos"]}

def merge_channel_direct(inp_list):
    return {"feat": torch.cat([x["feat"].clone() for x in inp_list], dim=1),
            "mask": torch.cat([x["mask"].clone() for x in inp_list], dim=0),
            "pos": torch.cat([x["pos"].clone() for x in inp_list], dim=1)}


def get_qkv(inp_list):
    """The 1st element of the inp_list is about the template,
    the 2nd (the last) element is about the search region"""
    dict_x = inp_list[-1]
    dict_c = {"feat": torch.cat([x["feat"] for x in inp_list], dim=0),
              "mask": torch.cat([x["mask"] for x in inp_list], dim=1),
              "pos": torch.cat([x["pos"] for x in inp_list], dim=0)}  # concatenated dict
    q = dict_x["feat"] + dict_x["pos"]
    k = dict_c["feat"] + dict_c["pos"]
    v = dict_c["feat"]
    key_padding_mask = dict_c["mask"]
    return q, k, v, key_padding_mask


def get_z_x(inp_list):
    """The 1st element of the inp_list is about the template,
    the 2nd (the last) element is about the search region"""

    z = inp_list[0]["feat"].permute(1, 0, 2)
    x = inp_list[1]["feat"].permute(1, 0, 2)

    return z, x
