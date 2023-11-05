# Redefining all the classes, functions, and data for execution.

class CelestialBody:
    def __init__(self, name, radius, children=None):
        self.name = name
        self.radius = radius
        self.children = children if children else []
        self.distance_from_parent = 0
        self.parent = None
        for child in self.children:
            child.parent = self

def calculate_precise_sphere_of_influence(body, separation):
    last_child_distance = 0
    
    # If the body has children, first calculate their distances
    if body.children:
        for child in body.children:
            child.distance_from_parent = last_child_distance + separation + child.radius
            last_child_distance = child.distance_from_parent + child.radius
            # Recursively calculate for children of the child
            last_child_distance = calculate_precise_sphere_of_influence(child, separation)
    
    # Update the radius of the parent body to include its furthest child's radius
    if body.children:
        body.radius = last_child_distance + body.children[-1].radius
    
    return body.radius + body.distance_from_parent

def print_reverted_distances(body, indent=0):
    if hasattr(body, 'parent') and body.parent:
        print("    " * indent + f"Distance from {body.name} to {body.parent.name} is {body.distance_from_parent}mm")
    if body.children:
        for child in body.children:
            print_reverted_distances(child, indent+1)

# Example Data and Calculation
sun = CelestialBody("Sun", 50, [
    CelestialBody("Mercury", 5),
    CelestialBody("Venus", 9),
    CelestialBody("Earth", 10, [CelestialBody("Moon", 4)]),
    CelestialBody("Mars", 6, [
        CelestialBody("Phobos", 1),
        CelestialBody("Deimos", 1.6)
    ]),
    CelestialBody("Barycentre", 0, [
        CelestialBody("Barycentre1", 0, [
            CelestialBody("Binary1", 5),
            CelestialBody("Binary2", 5)
        ]),
        CelestialBody("Barycentre2", 0, [
            CelestialBody("Binary3", 5),
            CelestialBody("Binary4", 5)
        ])
    ])
])

separation = 5
calculate_precise_sphere_of_influence(sun, separation)

# Print the final precise distances
corrected_output_v2 = print_reverted_distances(sun)
corrected_output_v2