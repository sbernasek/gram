from copy import deepcopy


class Mutation:
    """
    Base class for gene regulatory network models that provides methods for generating mutant genotypes.
    """

    @classmethod
    def remove_perturbed_reactions(cls, cell):
        """ Remove perturbed reactions and/or repressors from a cell. """

        # define filter function
        unaffected = lambda x: not x.perturbed

        # filter perturbed reactions
        reactions = list(filter(unaffected, cell.reactions))

        # if repressors are perturbed, remove them
        for rxn in reactions:
            if rxn.type == 'Hill':
                rxn.repressors = list(filter(unaffected, rxn.repressors))

        # assign filtered reactions to cell
        cell.reactions = reactions

    def perturb(self):
        """ Returns copy of cell without perturbed reactions/repressors. """
        cell = deepcopy(self)
        self.remove_perturbed_reactions(cell)
        return cell
