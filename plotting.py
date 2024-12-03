import matplotlib.pyplot as plt
import numpy as np
from signalearn.learning import reduce
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd

def plot_probability_scatter(result, n_components = 2):
    fig = plt.figure()
    
    probabilities = np.vstack(result.class_probabilities)
    labels = np.array(result.labels)

    if len(result.classes) > 2: 
        
        probabilities = reduce(probabilities, n_components=n_components)
        if n_components == 2:
            # 2D Plot
            scatter = plt.scatter(probabilities[:, 0], probabilities[:, 1], c=labels, cmap='viridis', alpha=0.6)
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title('2D Projection of Class Probabilities')
            plt.colorbar(scatter, label='Class Labels')

        elif n_components == 3:
            # 3D Plot
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(probabilities[:, 0], probabilities[:, 1], probabilities[:, 2], 
                                c=labels, cmap='viridis', alpha=0.6)
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')
            ax.set_title('3D Projection of Class Probabilities')
            fig.colorbar(scatter, label='Class Labels')

    elif len(result.classes) == 2:

        indices = np.arange(probabilities.shape[0])
        np.random.shuffle(indices)

        probabilities_shuffled = probabilities[indices]
        labels_shuffled = labels[indices]

        red_blue_cmap = ListedColormap(["blue", "red"])
        scatter = plt.scatter(
            np.arange(len(probabilities_shuffled[:, 1])), 
            probabilities_shuffled[:, 1], 
            c=labels_shuffled, 
            cmap=red_blue_cmap,
            alpha=0.6, 
            label=result.classes[1]
        )

        plt.xlabel('Scan')
        plt.ylabel(f'Probability of {result.classes[1]}')
        plt.title('Class Probabilities')

        plt.gca().xaxis.set_visible(False)
        plt.tick_params(axis='x', which='both', bottom=False, top=False)

    handles, _ = scatter.legend_elements()
    class_legend = plt.legend(handles, result.classes, title="Classes", loc="upper right")
    plt.gca().add_artist(class_legend)

    plt.savefig(f"results/probabilities-scatter-{result.name}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_probability_distribution_binary(result, class_name='Class', label_mapping=None):
    plt.figure()
    
    probabilities = np.vstack(result.class_probabilities)
    labels = np.array(result.labels)

    decoded_labels = np.array([result.classes[label] for label in labels])

    if label_mapping is not None:
        mapped_labels = np.array([label_mapping.get(label, f"Unknown ({label})") for label in decoded_labels])
        labels_unique = [label_mapping.get(label, f"Unknown ({label})") for label in result.classes]
    else:
        mapped_labels = decoded_labels
        labels_unique = result.classes

    data = pd.DataFrame({
        "Probability": probabilities[:, 1],
        class_name: mapped_labels
    })

    sns.kdeplot(data=data, x="Probability", hue=class_name, hue_order=labels_unique, common_norm=False, fill=True, alpha=0.5)

    plt.xlim(-0.1, 1.1)
    plt.gca().set_xticks(np.linspace(0, 1, 5))
    plt.gca().set_xticklabels([f'{tick:.2f}' for tick in np.linspace(0, 1, 5)])

    plt.gca().tick_params(axis='y', which='both', left=False, labelleft=False)

    plt.xlabel(f"Probability of {class_name + ' ' if class_name != '' else ''}{labels_unique[1]}")
    plt.ylabel('Probability Density')
    plt.title(f'{class_name} Probability Distribution')

    plt.savefig(f"results/probabilities-distribution-{result.name}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_probability_distribution_multiclass(result):
    plt.figure()

    probabilities = np.vstack(result.class_probabilities)
    labels = np.array(result.labels)

    data = pd.DataFrame(probabilities, columns=result.classes)
    data['True Label'] = [result.classes[label] for label in labels]

    data_melted = data.melt(
        id_vars='True Label',
        var_name='Predicted Class',
        value_name='Probability'
    )

    sns.kdeplot(
        data=data_melted,
        x="Probability",
        hue="Predicted Class",
        # multiple="stack",  # Show stacked KDEs for better separation
        common_norm=False,
        fill=True,
        alpha=0.6
    )

    plt.xlim(-0.1, 1.1)
    plt.gca().set_xticks(np.linspace(0, 1, 5))
    plt.gca().set_xticklabels([f'{tick:.2f}' for tick in np.linspace(0, 1, 5)])
    plt.gca().tick_params(axis='y', which='both', left=False, labelleft=False)

    plt.xlabel('Predicted Probability')
    plt.ylabel('Probability Density')
    plt.title('Class Probability Distribution')

    plt.savefig(f"results/probabilities-distribution-{result.name}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_importances(result):
    plt.figure()

    best_index = result.scores.index(max(result.scores))

    plt.plot(result.q_range, result.feature_importances[best_index])

    plt.xlabel(r'$Q$ ($\AA^{-1}$)')
    plt.ylabel('Importance')
    plt.title('Feature Importances')

    plt.savefig(f"results/importances-{result.name}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_pca(points, label=None, n_components=2):
    if n_components not in [2, 3]:
        raise ValueError("n_components must be either 2 or 3.")
    y = [point.y for point in points]
    labels = [getattr(point, label) for point in points] if label else None

    if labels:
        unique_labels = list(set(labels))
        label_to_color = {label: idx for idx, label in enumerate(unique_labels)}
        colors = [label_to_color[label] for label in labels]
        cmap = plt.cm.get_cmap('viridis', len(unique_labels))
    else:
        colors = None
        cmap = None

    y_reduced = reduce(y, n_components)  # Assuming reduce reduces dimensions to 2 or 3 as needed.

    if n_components == 2:
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            y_reduced[:, 0],
            y_reduced[:, 1],
            alpha=0.7,
            edgecolors='k',
            c=colors,
            cmap=cmap
        )
        ax.set_title("Principal Component Analysis (2D)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.grid(True)

    elif n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            y_reduced[:, 0],
            y_reduced[:, 1],
            y_reduced[:, 2],
            alpha=0.7,
            edgecolors='k',
            c=colors,
            cmap=cmap
        )
        ax.set_title("Principal Component Analysis (3D)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")

    if label is not None and labels:
        legend1 = ax.legend(
            handles=[plt.Line2D([0], [0], marker='o', color='w', label=unique_label,
                                markersize=10, markerfacecolor=cmap(idx / len(unique_labels)))
                     for idx, unique_label in enumerate(unique_labels)],
            title="Labels"
        )
        ax.add_artist(legend1)

    plt.show()

def plot_point(point):
    plt.figure()

    plt.xlabel(f'{point.xlabel} ({point.xunit})')
    plt.ylabel(f'{point.ylabel}')
    plt.plot(point.x, np.log1p(point.y))

    plt.title(point.title)

    plt.show()