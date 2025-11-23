# update_subreddits.py
import json
import os

CONFIG_PATH = "config/subreddits.json"

def load_subreddits():
    if not os.path.exists(CONFIG_PATH):
        print("⚠️  Config file not found. Creating a new one...")
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_subreddits(data):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print("✅ Subreddit list updated successfully!")

def list_subreddits(data):
    print("\n📂 Current subreddit categories:\n")
    for category, subs in data.items():
        print(f"[{category}]")
        for s in subs:
            print(f"  - {s}")
        print()

def add_subreddit(data):
    category = input("Enter category name (e.g., Law, Construction, Tech): ").strip()
    subreddit = input("Enter subreddit name (without r/): ").strip()

    if category not in data:
        data[category] = []
    if subreddit not in data[category]:
        data[category].append(subreddit)
        print(f"✅ Added r/{subreddit} under {category}.")
    else:
        print("⚠️ That subreddit already exists in this category.")

def remove_subreddit(data):
    category = input("Enter category: ").strip()
    if category not in data:
        print("⚠️ Category not found.")
        return

    subreddit = input("Enter subreddit name to remove: ").strip()
    if subreddit in data[category]:
        data[category].remove(subreddit)
        print(f"🗑️ Removed r/{subreddit} from {category}.")
    else:
        print("⚠️ Subreddit not found in that category.")

def edit_category_name(data):
    old_cat = input("Enter the existing category name: ").strip()
    if old_cat not in data:
        print("⚠️ Category not found.")
        return
    new_cat = input("Enter new category name: ").strip()
    data[new_cat] = data.pop(old_cat)
    print(f"✅ Renamed category '{old_cat}' → '{new_cat}'.")

def main():
    data = load_subreddits()

    while True:
        print("\n===== Subreddit Manager =====")
        print("1. List subreddits")
        print("2. Add subreddit")
        print("3. Remove subreddit")
        print("4. Edit category name")
        print("5. Save & Exit")
        print("==============================")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            list_subreddits(data)
        elif choice == "2":
            add_subreddit(data)
        elif choice == "3":
            remove_subreddit(data)
        elif choice == "4":
            edit_category_name(data)
        elif choice == "5":
            save_subreddits(data)
            break
        else:
            print("⚠️ Invalid choice. Try again.")

if __name__ == "__main__":
    main()
