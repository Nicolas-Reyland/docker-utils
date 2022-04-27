#!/usr/bin/env python3.10
from __future__ import annotations
from abc import ABC, abstractmethod
from argparse import ArgumentParser
import os
import sys
import json
import docker
import humanize
import re

# Globals
VERSION_TAG_REGEX_PATTERN = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")
DOCKER_COMMANDS_LIST = list()


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

# Rmoi parser
imgf_parser = sub_parsers.add_parser(
    "rmoi",
    description="Remove old images",
    help="Remove old images. This will never delete the latest version of an image",
)
imgf_parser.add_argument(
    "image_name",
    metavar="image-name",
    help="Name of the image repository (exact match)",
)
imgf_parser.add_argument(
    "-M",
    "--rm-major",
    action="store_true",
    help="Remove all images which are tagged with a different major version than the latest one",
)
imgf_parser.add_argument(
    "-m",
    "--rm-minor",
    action="store_true",
    help="Remove all images which are tagged with a different minor version than the latest one",
)
imgf_parser.add_argument(
    "-p",
    "--rm-patch",
    action="store_true",
    help="Remove all images which are tagged with a different patch version than the latest one (Basically removes all non-magic images)",
)
imgf_parser.add_argument(
    "-l",
    "--keep-last",
    type=int,
    default=0,
    help="Keep the last n images. Remove all the rest (based on creation date). A value of zero means no removal",
)
imgf_parser.add_argument(
    "-o",
    "--rm-old",
    type=int,
    default=0,
    help="Remove the n oldest images. Keep the rest (based on creation date). A value of zero means no removal",
)
imgf_parser.add_argument(
    "-f",
    "--force",
    dest="rmoi_force",
    action="store_true",
    help="Force removal of the images",
)
imgf_parser.add_argument(
    "--no-prune",
    action="store_true",
    help="Do not delete untagged parents",
)
imgf_parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print the images that would be removed, without actually removing those",
)


# Utils
def print_warning(*args, **kwargs):
    print("Warning: ", end="")
    kwargs |= {
        "flush": True,
    }
    print(*args, **kwargs)

def print_error(*args, exit_program: bool=True, exit_code: int=1, **kwargs):
    print("Error: ", file=sys.stderr, end="")
    kwargs |= {
        "file": sys.stderr,
        "flush": True,
    }
    print(*args, **kwargs)
    if exit_program:
        sys.exit(exit_code)

def register_command(command_name: str):
    global DOCKER_COMMANDS_LIST
    def class_wrapper(command_class):
        assert issubclass(command_class, DockerCommandBase)
        instance = command_class(command_name)
        DOCKER_COMMANDS_LIST.append(instance)
    return class_wrapper

def extract_tag_from_full(full_tag: str) -> str:
    *_, tag = full_tag.rpartition(":")
    return tag

def extract_repo_from_full(full_tag: str) -> str:
    repo, *_ = full_tag.rpartition(":")
    return repo

def extract_repo_and_tag_from_full(full_tag: str) -> str:
    repo, _, tag = full_tag.rpartition(":")
    return repo, tag

def extract_short_id(full_short_id: str) -> str:
    *_, short_id = full_short_id.partition(":")
    return short_id

def extract_valid_tag(image_short_tags: list[str]) -> str:
    valid_tags = set(filter(
        VERSION_TAG_REGEX_PATTERN.match,
        image_short_tags,
    ))
    assert len(valid_tags) == 1, f"Image has multiple versions: {', '.join(valid_tags)}"
    return valid_tags.pop()

def extract_version_from_image_short_tags(image_short_tags: list[str]) -> tuple[int]:
    tag = extract_valid_tag(image_short_tags)
    tag = tag[1:] # remove 'v'
    tag_atoms = tag.split(".")
    return tuple(map(int, tag_atoms))

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

def image_in_version_scope(latest_version: tuple[int], short_tags: list[str], rm_major: bool, rm_minor: bool, rm_patch: bool) -> bool:
    version = extract_version_from_image_short_tags(short_tags)
    if rm_patch:
        return version != latest_version
    if rm_minor:
        return version[1] != latest_version[1] or version[0] != latest_version[0]
    if rm_major:
        return version[0] != latest_version[0]
    return False

def image_str(image) -> str:
    short_tags = set(map(extract_tag_from_full, image.tags))
    # basically take the shortest repo name and remove the tag from it
    repo = extract_repo_from_full(sorted(image.tags, key=lambda full_tag: len(extract_repo_from_full(full_tag)))[0])
    tag = extract_valid_tag(short_tags)
    short_tags.remove(tag)
    return f"Image {repo}:{tag}{(' (' + ', '.join(short_tags) + ')') if short_tags else ''}"


# Commands
class DockerCommandBase(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def execute(self, client, args: list[str]) -> None:
        """Execute the arguments"""


@register_command("build")
class DockerBuild(DockerCommandBase):
    def __init__(self, name: str):
        super().__init__(name)
        self.default_image_tag = "v1.0.0"
        self.default_upgrade_rule = "patch"
        self.default_push_rule = "local"
        self.magic_tags = ["latest", "dev"]

    def execute(self, client, args: list[str]) -> None:
        """Disclaimer : I use a 'vMAJOR.MINOR.PATCH' tagging convention for my docker images
        """
        # docker-utils dev build image-name major|minor|patch|none push|push-tag-only|local|
        low_level_client = docker.APIClient()
        if args.dev:
            if args.upgrade != "none":
                print_warning(f"upgrade policy is '{args.upgrade}', but will be ignored due to development mode")
            if args.tag:
                new_image_tag = args.tag
                print_warning("setting tag manually when in development mode")
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
                    print_error(f"This image has no non-magic tag. Magic tags: {', '.join(self.magic_tags)}", exit_program=True)
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
                            print_error(f"Should never reach this point. Unkown upgrade policy: {args.upgrade}", exit_program=True)
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


@register_command("imgf")
class DockerImgf(DockerCommandBase):
    def __init__(self, name: str):
        super().__init__(name)
        self.repo_width = 56
        self.tag_width = 15
        self.short_id_width = 20
        self.size_width = 10

    def execute(self, client, args: list[str]) -> None:
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
                short_id = extract_short_id(image.short_id)
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


@register_command("rmoi")
class DockerRemoveOldImages(DockerCommandBase):
    def __init__(self, name: str):
        super().__init__(name)

    def execute(self, client, args: list[str]) -> None:
        """Will only execute on images that are named after tag convention
        """
        # Dry run announcment
        if args.dry_run:
            print("Dry run for rmoi")
        # Check for options compatibility
        if args.dev:
            print_error("Command 'rmoi' has no development mode. Aborting", exit_program=True)
        if args.keep_last != 0:
            assert args.rm_old == 0, \
                "Cannot use -l/--keep-last option with -o/--rm-old"
            assert not args.rm_major and not args.rm_minor and not args.rm_patch, \
                "Cannot use -l/--rm-old option with any version-filtering option -M/--rm-major/-m/--rm-minor/-p/--rm-patch"
        if args.rm_old != 0:
            assert args.keep_last == 0, \
                "Cannot use -o/--rm-old option with -l/--keep-last"
            assert not args.rm_major and not args.rm_minor and not args.rm_patch, \
                "Cannot use -l/--rm-old option with any version-filtering option -M/--rm-major/-m/--rm-minor/-p/--rm-patch"
        # Gather images
        images = client.images.list(name=args.image_name)
        for image in images:
            image.short_tags = list(map(extract_tag_from_full, image.tags))
        # remove all images that don't respect tagging convention from list (and dev-only if in it)
        images = filter(
            lambda image: any(
                VERSION_TAG_REGEX_PATTERN.match(tag)
                for tag in image.short_tags
                ),
            images)
        # sort by creation date. latest is first. oldest is last
        images = sorted(images, key=lambda image: image.attrs["Created"], reverse=True)
        latest_image = images.pop(0)
        assert any(tag == "latest" for tag in latest_image.short_tags), \
            f"Latest image (by creation date) is not tagged 'latest': {', '.join(latest_image.short_tags)}"
        print(f"Safeguarding latest image: {image_str(latest_image)}")
        # Filter out of the list 'images' the ones that should be kept
        latest_version = extract_version_from_image_short_tags(latest_image.short_tags)
        # Version-based filtering
        if args.rm_patch or args.rm_minor or args.rm_major:
            images = list(filter(
                lambda image: image_in_version_scope(
                    latest_version,
                    image.short_tags,
                    args.rm_major,
                    args.rm_minor,
                    args.rm_patch
                ),
                images,
            ))
        # Keep last n images
        if args.keep_last != 0:
            images = images[args.keep_last:]
        # Remove olest n images
        if args.rm_old != 0:
            images = images[-args.rm_old:]

        for image in images:
            short_id = extract_short_id(image.short_id)
            print(f"Removing {image_str(image)} with short id {short_id}")
            if not args.dry_run:
                client.images.remove(
                    image=short_id,
                    force=args.rmoi_force,
                    noprune=args.no_prune,
                )
        if args.dry_run:
            print("Dry-run finished")


# Main function
def main(client):
    global DOCKER_COMMANDS_LIST
    args = parser.parse_args()
    for command in DOCKER_COMMANDS_LIST:
        if command.name == args.command:
            command.execute(client, args)
            exit(0)

if __name__ == "__main__":
    client = docker.from_env()
    main(client)
