from genessa.models.twostate import TwoStateCell
from .mutation import Mutation


class TwoStateModel(TwoStateCell, Mutation):
    """
    Class defines a cell with a single protein coding gene subject to negative feedback. Transcription is based on a two-state model.

    Attributes:

        name (str) - name of controlled gene

    Inherited Attributes:

        off_states (dict) - {name: node_id} pairs

        on_states (dict) - {name: node_id} pairs

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

    def __init__(self, name='X', k0=1, k1=1, k2=1, g0=1, g1=1, g2=1):
        """
        Instantiate the two state model.

        Args:

            name (str) - name of controlled gene

            k0 (float) - gene activation rate constant

            k1 (float) - transcription rate constant

            k2 (float) - translation rate constant

            g0 (float) - gene deactivation rate constant

            g1 (float) - transcript decay rate constant

            g2 (float) - protein decay rate constant

        """

        self.name = name

        # instantiate twostate cell with a single gene
        gene_kw = dict(k0=0, k1=k1, k2=k2, g0=g0, g1=g1, g2=g2)
        super().__init__(genes=(self.name,), I=1, **gene_kw)

        # add transcriptional activation by input
        self.add_activation(gene=self.name, activator='IN', k=k0)

    def add_transcriptional_feedback(self,
                                     k=None,
                                     atp_sensitive=2,
                                     ribosome_sensitive=True,
                                     **kwargs):
        """
        Adds transcriptional auto-repression.

        Args:

            k (float) - rate parameter (feedback strength)

            atp_sensitive (bool) - scale rate parameter with metabolism

            ribosome_sensitive (bool) - scale rate parameter with ribosomes

            kwargs: keyword arguments for reaction

        """

        self.add_transcriptional_repressor(
             actuator=self.name,
             target=self.name,
             k=k,
             atp_sensitive=atp_sensitive,
             ribosome_sensitive=ribosome_sensitive,
             **kwargs)

    def add_post_transcriptional_feedback(self,
                                     k=None,
                                     atp_sensitive=2,
                                     ribosome_sensitive=True,
                                     **kwargs):
        """
        Adds linear negative feedback applied to transcript level.

        Args:

            k (float) - rate parameter (feedback strength)

            atp_sensitive (int) - order of metabolism dependence

            ribosome_sensitive (bool) - scale rate parameter with ribosomes

            kwargs: keyword arguments for reaction

        """

        self.add_linear_feedback(
             sensor=self.name,
             target=self.name,
             mode='transcript',
             k=k,
             atp_sensitive=atp_sensitive,
             ribosome_sensitive=ribosome_sensitive,
             **kwargs)

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
