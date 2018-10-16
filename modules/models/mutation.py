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

    def add_feedback(self, eta0, eta1, eta2, perturbed=False):
        """
        Add feedback at the gene, transcript, and protein levels.

        Args:

            eta0 (float) - transcriptional feedback strength

            eta1 (float) - post-transcriptional feedback strength

            eta2 (float) - post-translational feedback strength

            perturbed (bool) - if True, feedback is sensitive to perturbation

        """
        self.add_transcriptional_feedback(k=eta0, perturbed=perturbed)
        self.add_post_transcriptional_feedback(k=eta1, perturbed=perturbed)
        self.add_post_translational_feedback(k=eta2, perturbed=perturbed)
