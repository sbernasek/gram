from genessa.models.simple import SimpleCell
from .mutation import Mutation


class SimpleModel(SimpleCell, Mutation):
    """
    Class defines a cell with a single protein state subject to negative feedback. All reaction rates are based on linear propensity functions.

    Attributes:

        name (str) - name of controlled gene

    Inherited Attributes:

        transcripts (dict) - {name: node_id} pairs

        proteins (dict) - {name: node_id} pairs

        phosphorylated (dict) - {name: node_id} pairs

        nodes (np.ndarray) - vector of node indices

        node_key (dict) - {state dimension: node id} pairs

        reactions (list) - list of reaction objects

        stoichiometry (np.ndarray) - stoichiometric coefficients, (N,M)

        N (int) - number of nodes

        M (int) - number of reactions

        I (int) - number of inputs

    """

    def __init__(self, name='X', k=1, g=1):
        """
        Instantiate a simple model of a single protein.

        Args:

            name (str) - name of controlled protein

            k (float) - protein synthesis rate constant

            g (float) - protein decay rate constant

        """

        self.name = name

        # instantiate linear cell with a single gene activated by the input
        gene_kw = dict(g=g)
        super().__init__(genes=(self.name,), I=1, **gene_kw)

        # add synthesis driven by input
        self.add_activation(protein=self.name, activator='IN', k=k)

    def add_post_translational_feedback(self,
            k=None,
            atp_sensitive=2,
            ribosome_sensitive=True,
            **kwargs):
        """
        Adds linear negative feedback applied to protein level.

        Args:

            k (float) - rate parameter (feedback strength)

            atp_sensitive (int) - order of metabolism dependence

            ribosome_sensitive (bool) - scale rate parameter with ribosomes

            kwargs: keyword arguments for reaction

        """

        self.add_linear_feedback(
             sensor=self.name,
             target=self.name,
             mode='protein',
             k=k,
             atp_sensitive=atp_sensitive,
             ribosome_sensitive=ribosome_sensitive,
             **kwargs)

    def add_feedback(self, eta, perturbed=False):
        """
        Add feedback.

        Args:

            eta (float) - feedback strength

            perturbed (bool) - if True, feedback is sensitive to perturbation

        """
        self.add_post_translational_feedback(k=eta, perturbed=perturbed)
