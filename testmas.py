from itertools import permutations

def count_transpositions_baryons(perm, group1, group2):
    transpositions = 1  # Start with a factor of +1

    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            # Check if the current pair involves a within-group swap
            if (i in group1 and j in group1) or (i in group2 and j in group2):
                if perm[i] > perm[j]:  # If out of order, it's a transposition
                    transpositions *= -1
            # Check if the current pair involves a between-group swap
            elif (i in group1 and j in group2) or (i in group2 and j in group1):
                if perm[i] > perm[j]:  # If out of order, it's a transposition
                    transpositions *= 1  # Keep the factor the same (implicitly +1)

    return transpositions

def unique_group_permutations_baryons(masses):
    # Group the indices by mass
    mass_dict = {}
    for idx, mass in enumerate(masses):
        if mass not in mass_dict:
            mass_dict[mass] = []
        mass_dict[mass].append(idx)

    # Generate permutations for each group of identical masses
    perms_per_group = {mass: list(permutations(indices)) for mass, indices in mass_dict.items() if len(indices) > 1}

    # Start with the identity permutation
    complete_perms = [list(range(len(masses)))]

    # Define the two baryon groups
    group1 = [0, 1, 2]  # m1, m2, m3
    group2 = [3, 4, 5]  # m4, m5, m6

    # Combine permutations for each group
    for mass, perm_group in perms_per_group.items():
        new_complete_perms = []
        for base_perm in complete_perms:
            for group_perm in perm_group:
                new_perm = base_perm.copy()
                for i, idx in enumerate(group_perm):
                    new_perm[mass_dict[mass][i]] = base_perm[idx]
                new_complete_perms.append(new_perm)
        complete_perms = new_complete_perms

    # Calculate antisymmetrization factors for each permutation, considering baryon groups
    antisym_factors = [count_transpositions_baryons(perm, group1, group2) for perm in complete_perms]

    return complete_perms, antisym_factors

# Example usage
masses = ['1.0', '2.0', '2.0', '1.0', '1.0', '2.0']
perms, antisym_factors = unique_group_permutations_baryons(masses)

print(f"Masses: {masses}")
print(len(perms), "unique permutations found.")

# Output permutations and corresponding antisymmetrization factors
for perm, factor in zip(perms, antisym_factors):
    print(f"Permutation: {perm}, Antisymmetrization Factor: {factor}")
