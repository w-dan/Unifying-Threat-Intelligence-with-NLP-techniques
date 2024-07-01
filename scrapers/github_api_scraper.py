import os
import requests
from typing import List
from dotenv import load_dotenv

def get_github_repo_commit_sha(owner: str, repo: str, branches: List[str], token: str) -> str:
    """
    Get the SHA of the latest commit on the specified branches of a GitHub repository.
    Checks if the master branch exists, if not, it checks the main branch.
    Could use any other branch name on the list in main.

    :param owner: Owner of the repository.
    :param repo: Name of the repository.
    :param branches: List of branches to fetch the commit SHA from.
    :param token: GitHub personal access token for authentication.
    :return: SHA of the latest commit and the valid branch name.
    """
    headers = {'Authorization': f'token {token}'}
    for branch in branches:
        url = f"https://api.github.com/repos/{owner}/{repo}/branches/{branch}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()['commit']['sha'], branch
        elif response.status_code in [401, 403, 404]:
            print(f"[-] Branch {branch} not found or access denied: {response.status_code}")
        else:
            print(f"[-] Error fetching commit SHA for branch {branch}: {response.status_code} - {response.text}")
            response.raise_for_status()
    
    raise ValueError("[?] None of the specified branches were found in the repository.")


def get_github_repo_tree(owner: str, repo: str, sha: str, token: str) -> List[dict]:
    """
    Get the entire tree of a GitHub repository.

    :param owner: Owner of the repository.
    :param repo: Name of the repository.
    :param sha: SHA of the commit to fetch the tree from.
    :param token: GitHub personal access token for authentication.

    :return: List of all files and directories in the repository.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{sha}?recursive=1"
    headers = {'Authorization': f'token {token}'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get('tree', [])
    else:
        print(f"[-] Error fetching repository tree: {response.status_code} - {response.text}")
        response.raise_for_status()


def download_file(url: str, local_path: str, token: str = None) -> None:
    """
    Download a file from a URL to a local path.

    :param url: URL of the file to download.
    :param local_path: Local path to save the downloaded file.
    :param token: GitHub personal access token for authentication.
    """
    headers = {'Authorization': f'token {token}'} if token else {}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open(local_path, 'wb') as file:
            file.write(response.content)
    else:
        print(f"[-] Error downloading file: {response.status_code} - {response.text}")
        response.raise_for_status()


def extract_pdfs_from_repo(owner: str, repo: str, local_dir: str, branches: List[str], token: str = None) -> None:
    """
    Extract all PDF files from a GitHub repository and save them locally.

    :param owner: Owner of the repository.
    :param repo: Name of the repository.
    :param local_dir: Local directory to save the PDF files.
    :param branches: List of branches to check for PDF files.
    :param token: GitHub personal access token for authentication.
    """
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    sha, valid_branch = get_github_repo_commit_sha(owner, repo, branches, token)
    tree = get_github_repo_tree(owner, repo, sha, token)

    for item in tree:
        if item['type'] == 'blob' and item['path'].endswith('.pdf'):
            file_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{valid_branch}/{item['path']}"
            print(f"\t ╰─[📜] Downloading {item['path']} from {file_url}")
            local_path = os.path.join(local_dir, os.path.basename(item['path']))
            download_file(file_url, local_path, token)


def extract_user_repo_from_url(url: str) -> tuple:
    """
    Extract the user and repo name from a GitHub URL.

    :param url: GitHub URL.
    :return: Tuple containing the user and repo name.
    """
    parts = url.split("/")
    user = parts[-2]
    repo = parts[-1]
    return user, repo



if __name__ == "__main__":
    load_dotenv(".env")
    TOKEN = os.getenv("GH_TOKEN")

    ##### example repo (WIP) but should work on any since it's recursive
    url = "https://github.com/CyberMonitor/APT_CyberCriminal_Campagin_Collections"
    local_dir = "./pdf_files"

    user, repo = extract_user_repo_from_url(url)

    extract_pdfs_from_repo(user, repo, local_dir, branches=["master", "main"], token=TOKEN)
