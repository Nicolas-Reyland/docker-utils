#!/usr/bin/env python3.10
from abc import ABC, abstractmethod
from argparse import ArgumentParser
import os
import json
import docker
import humanize
import re

# Globals
VERSION_TAG_REGEX_PATTERN = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")


# Argument parsing
parser = ArgumentParser(description="Docker-Utils, a tool for fast container and image management")
parser.add_argument(
    "--dev",
    action="store_true",
    help="Use development environment",
)

sub_parsers = parser.add_subparsers(
    help="Sub commands",
    dest="command",
)

# Build parser
build_parser = sub_parsers.add_parser(
    "build",
    description="Build docker images",
    help="Build docker images",
)
build_parser.add_argument(
    "image_name",
    metavar="image-name",
    help="Docker image name",
)
build_parser.add_argument(
    "upgrade",
    choices=["major", "minor", "patch", "none"],
    nargs="?",
    default="patch",
    help="Version number upgrade policy",
)
build_parser.add_argument(
    "push",
    choices=["push", "push-tag-only", "local"],
    nargs="?",
    default="push-tag-only",
    help="Image push policy",
)
build_parser.add_argument(
    "-t",
    "--tag",
    help="Manually set the tag string",
)
build_parser.add_argument(
    "-b",
    "--build-args",
    default=r"{}",
    help="Docker build arguments in json format",
)

# Imgf parser
imgf_parser = sub_parsers.add_parser(
    "imgf",
    description="List and filter docker images",
    help="List and filter docker images",
)
imgf_parser.add_argument(
    "name",
    help="Name or part of name of image",
)
imgf_parser.add_argument(
    "-e",
    "--exact-name",
    action="store_true",
    help="The name should be an exact match",
)
imgf_parser.add_argument(
    "-a",
    "--all",
    action="store_true",
    help="Also list image layers",
)
imgf_parser.add_argument(
    "-d",
    "--dangling",
    action="store_true",
    help="Also list dangling images",
)
imgf_parser.add_argument(
    "-u",
    "--unique",
    action="store_true",
    help="Only one image name per id",
)
imgf_parser.add_argument(
    "-s",
    "--space",
    action="store_true",
    help="Print an empty line between new image ids",
)


# Utils
def extract_tag_from_full(full_tag: str) -> str:
    *_, tag = full_tag.rpartition(":")
    return tag

def extract_repo_from_full(full_tag: str) -> str:
    repo, *_ = full_tag.rpartition(":")
    return repo

def extract_repo_and_tag_from_full(full_tag: str) -> str:
    repo, _, tag = tag.rpartition(":")
    return repo, tag

def upgrade_tag_major(tag: str) -> str:
    assert VERSION_TAG_REGEX_PATTERN.match(tag)
    tag = tag[1:] # remove 'v'
    major, *_ = tag.partition(".")
    return f"v{int(major) + 1}.0.0"

def upgrade_tag_minor(tag: str) -> str:
    assert VERSION_TAG_REGEX_PATTERN.match(tag)
    vmajor, minor, patch = tag.split(".")
    return f"{vmajor}.{int(minor) + 1}.0"

def upgrade_tag_patch(tag: str) -> str:
    assert VERSION_TAG_REGEX_PATTERN.match(tag)
    rest, _, patch = tag.rpartition(".")
    return f"{rest}.{int(patch) + 1}"


# Commands
class DockerCommandBase(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def execute(self, args: list[str]) -> None:
        """Execute the arguments"""

class DockerBuild(DockerCommandBase):
    def __init__(self, name: str):
        super().__init__(name)
        self.default_image_tag = "v1.0.0"
        self.default_upgrade_rule = "patch"
        self.default_push_rule = "local"
        self.magic_tags = ["latest", "dev"]

    def execute(self, client, args):
        # docker-utils dev build image-name major|minor|patch|none push|push-tag-only|local|
        low_level_client = docker.APIClient()
        if args.dev:
            if args.upgrade != "none":
                print(f"Warning: upgrade policy is '{args.upgrade}', but will be ignored due to development mode")
            if args.tag:
                new_image_tag = args.tag
                print("Warning: setting tag manually when in development mode")
            else:
                new_image_tag = "dev"
                print("Development mode: tag is set to 'dev'")
        else:
            old_images = client.images.list(name = args.image_name)
            image_tag = self.default_image_tag if not args.tag else args.tag
            if not old_images:
                print(f"No image found with this name. Starting with new tag {image_tag}")
                assert args.upgrade == "none", f"Cannot upgrade the version number when no other image with this name exists. Must use 'none' in this case."
            else:
                try:
                    latest_image = next(filter(lambda image: any(tag.endswith("latest") for tag in image.tags), old_images))
                except StopIteration:
                    print("No image with tag 'latest'. Using the most recent one instead.")
                    latest_image = sorted(old_images, key=lambda image: image.attrs["Created"], reverse=True)[0]
                all_tags = set(map(extract_tag_from_full, latest_image.tags))
                print(f"Detected latest version of {args.image_name} as having the tag{'s' if len(latest_image.tags) > 1 else ''} {', '.join(all_tags)}")
                try:
                    latest_image_tag = next(filter(lambda tag: tag not in self.magic_tags, all_tags))
                except StopIteration:
                    raise ValueError(f"This image has no non-magic tag. Magic tags: {', '.join(self.magic_tags)}")  
                # Updating tag
                if args.tag:
                    new_image_tag = args.tag
                else:
                    match args.upgrade:
                        case "none":
                            new_image_tag = latest_image_tag
                        case "major":
                            new_image_tag = upgrade_tag_major(latest_image_tag)
                        case "minor":
                            new_image_tag = upgrade_tag_minor(latest_image_tag)
                        case "patch":
                            new_image_tag = upgrade_tag_patch(latest_image_tag)
                        case _:
                            raise RuntimeError(f"Should never reach this point. Unkown upgrade policy: {args.upgrade}")
                print(f"Image tag {latest_image_tag} -> {new_image_tag}")
        full_new_image_name = f"{args.image_name}:{new_image_tag}"
        print(f"Final image name: {full_new_image_name}")

        kwargs = {
            "path": os.getcwd(),
            "tag": full_new_image_name,
            "network_mode": "host",
        }
        # Merge with custom build args
        kwargs |= json.loads(args.build_args)
        for line in low_level_client.build(**kwargs):
            data = json.loads(line)
            if "stream" in data:
                print(f"BUILD: {data['stream']}", end="")
            if "aux" in data:
                print(f"AUX: {json.dumps(data, indent=4)}")


class DockerImgf(DockerCommandBase):
    def __init__(self, name: str):
        super().__init__(name)
        self.repo_width = 56
        self.tag_width = 15
        self.short_id_width = 20
        self.size_width = 10

    def execute(self, client, args):
        # find the images
        f_kwargs = {
            "name": args.name if args.exact_name else None,
            "all": args.all,
            "filters": {
                "dangling": args.dangling,
            }
        }
        images = client.images.list(**f_kwargs)
        if not args.exact_name:
            images = filter(lambda image: any(args.name in tag for tag in image.tags), images)
        # print the images
        self._print_fixed_width("REPOSITORY", self.repo_width)
        self._print_fixed_width("TAG", self.tag_width)
        self._print_fixed_width("IMAGE ID", self.short_id_width)
        self._print_fixed_width("SIZE", self.size_width)
        print()
        for image in images:
            self._print_image(image, unique=args.unique)
            if args.space:
                print()

    def _print_image(self, image, unique=False):
        if unique:
            tags = [sorted(image.tags, key=lambda tag: len(tag))[0]]
        else:
            tags = image.tags
        for i,full_tag in enumerate(tags):
            repo, tag = extract_repo_and_tag_from_full(full_tag)
            self._print_fixed_width(repo, self.repo_width)
            self._print_fixed_width(tag, self.tag_width)
            if i == 0:
                *_, short_id = image.short_id.partition(":")
                size = humanize.naturalsize(image.attrs["Size"])
            else:
                short_id = size = ""
            self._print_fixed_width(short_id, self.short_id_width)
            self._print_fixed_width(size, self.size_width)
            print()

    def _print_fixed_width(self, content, width, char = " ", **kwargs):
        content_width = len(content)
        num_spaces = max(width - content_width, 0)
        string = content + char * num_spaces
        print(string, end="", **kwargs)


def main(client):
    commands: list[DockerCommandBase] = list()
    commands.append(DockerBuild("build"))
    commands.append(DockerImgf("imgf"))
    args = parser.parse_args()
    for command in commands:
        if command.name == args.command:
            command.execute(client, args)

if __name__ == "__main__":
    client = docker.from_env()
    main(client)
