import os
import matplotlib.pyplot as plt
from PIL import Image
import random
import os

dataset_path = r"C:\Users\anura\Downloads\archive\Garbage classification\Garbage classification"

if not os.path.exists(dataset_path):
    print(
        f"Folder '{dataset_path}' not found. Please create it and add your images.")
else:
    print(f"Great! Found the folder: {dataset_path}")


def count_images_per_category(dataset_path):
    print("\nCounting images in each category...")

    categories = os.listdir(dataset_path)

    for category in categories:
        category_path = os.path.join(dataset_path, category)

        if not os.path.isdir(category_path):
            continue

        images = [f for f in os.listdir(category_path)
                  if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))]

        print(f"{category}: {len(images)} images")

    return categories


def show_sample_images(dataset_path, num_samples=6):
    categories = [d for d in os.listdir(dataset_path)
                  if os.path.isdir(os.path.join(dataset_path, d))]

    if not categories:
        print("No category folders found. Make sure your dataset is organized!")
        return

    plt.figure(figsize=(12, 8))

    for i in range(min(num_samples, len(categories) * 2)):
        category = random.choice(categories)
        category_path = os.path.join(dataset_path, category)

        images = [f for f in os.listdir(category_path)
                  if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))]

        if not images:
            continue

        random_image = random.choice(images)
        image_path = os.path.join(category_path, random_image)

        plt.subplot(2, 3, i + 1)
        img = Image.open(image_path)
        plt.imshow(img)
        plt.title(f"Category: {category}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('sample_images.png')
    print("\nSaved sample images to 'sample_images.png'")
    plt.show()


def check_image_sizes(dataset_path, num_to_check=20):
    import matplotlib.pyplot as plt

    dataset_path = r"C:\Users\anura\Downloads\archive\Garbage classification\Garbage classification"

    if not os.path.exists(dataset_path):
        print(
            f"Folder '{dataset_path}' not found. Please create it and add your images.")
    else:
        print(f"Great! Found the folder: {dataset_path}")

    print("\nChecking image dimensions...")

    categories = [d for d in os.listdir(dataset_path)
                  if os.path.isdir(os.path.join(dataset_path, d))]

    sizes = []

    for category in categories:
        category_path = os.path.join(dataset_path, category)
        images = [f for f in os.listdir(category_path)
                  if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))]

        for img_name in images[:5]:
            img_path = os.path.join(category_path, img_name)
            img = Image.open(img_path)
            sizes.append(img.size)

    print(f"Sample image sizes: {sizes[:10]}")

    if sizes:
        avg_width = sum([s[0] for s in sizes]) / len(sizes)
        avg_height = sum([s[1] for s in sizes]) / len(sizes)
        print(f"Average size: {int(avg_width)} x {int(avg_height)} pixels")


if __name__ == "__main__":

    categories = count_images_per_category(dataset_path)
    show_sample_images(dataset_path)
    check_image_sizes(dataset_path)
