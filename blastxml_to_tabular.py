#!/usr/bin/env python
"""Convert a BLAST XML file to tabular output.

Designed to convert BLAST XML files into tabular BLAST output (either
std for standard 12 columns, or ext for the extended 25 columns offered
in the Galaxy BLAST+ wrappers).

The 12 columns output are 'qseqid sseqid pident length mismatch gapopen qstart
qend sstart send evalue bitscore' or 'std' at the BLAST+ command line, which
mean:

====== ========= ============================================
Column NCBI name Description
------ --------- --------------------------------------------
     1 qseqid    Query Seq-id (ID of your sequence)
     2 sseqid    Subject Seq-id (ID of the database hit)
     3 pident    Percentage of identical matches
     4 length    Alignment length
     5 mismatch  Number of mismatches
     6 gapopen   Number of gap openings
     7 qstart    Start of alignment in query
     8 qend      End of alignment in query
     9 sstart    Start of alignment in subject (database hit)
    10 send      End of alignment in subject (database hit)
    11 evalue    Expectation value (E-value)
    12 bitscore  Bit score
====== ========= ============================================

The additional columns offered in the Galaxy BLAST+ wrappers are:

====== ============= ===========================================
Column NCBI name     Description
------ ------------- -------------------------------------------
    13 sallseqid     All subject Seq-id(s), separated by ';'
    14 score         Raw score
    15 nident        Number of identical matches
    16 positive      Number of positive-scoring matches
    17 gaps          Total number of gaps
    18 ppos          Percentage of positive-scoring matches
    19 qframe        Query frame
    20 sframe        Subject frame
    21 qseq          Aligned part of query sequence
    22 sseq          Aligned part of subject sequence
    23 qlen          Query sequence length
    24 slen          Subject sequence length
    25 salltitles    All subject titles, separated by '&lt;&gt;'
====== ============= ===========================================

Most of these fields are given explicitly in the XML file, others some like
the percentage identity and the number of gap openings must be calculated.

Be aware that the sequence in the extended tabular output or XML direct from
BLAST+ may or may not use XXXX masking on regions of low complexity. This
can throw the off the calculation of percentage identity and gap openings.
[In fact, both BLAST 2.2.24+ and 2.2.25+ have a subtle bug in this regard,
with these numbers changing depending on whether or not the low complexity
filter is used.]

This script attempts to produce identical output to what BLAST+ would have done.
However, check this with "diff -b ..." since BLAST+ sometimes includes an extra
space character (probably a bug).
"""

from __future__ import print_function

import os
import re
import sys
from optparse import OptionParser

if "-v" in sys.argv or "--version" in sys.argv:
    print("v0.2.01")
    sys.exit(0)

if sys.version_info[:2] >= (2, 5):
    try:
        from xml.etree import cElementTree as ElementTree
    except ImportError:
        from xml.etree import ElementTree as ElementTree
else:
    from galaxy import eggs  # noqa - ignore flake8 F401
    import pkg_resources

    pkg_resources.require("elementtree")
    from elementtree import ElementTree

if len(sys.argv) == 4 and sys.argv[3] in ["std", "x22", "ext"]:
    # False positive if user really has a BLAST XML file called 'std' or 'ext'...
    sys.exit(
        """ERROR: The script API has changed, sorry.

Instead of the old style:

$ python blastxml_to_tabular.py input.xml output.tabular std

Please use:

$ python blastxml_to_tabular.py -o output.tabular -c std input.xml

For more information, use:

$ python blastxml_to_tabular.py -h
"""
    )

usage = """usage: %prog [options] blastxml[,...]

Convert one (or more) BLAST XML files into a single tabular file.

The columns option can be 'std' (standard 12 columns), 'ext'
(extended 25 columns), or a list of BLAST+ column names like
'qseqid,sseqid,pident' (space or comma separated).

Note if using a list of column names, currently ONLY the 25
extended column names are supported.
"""
parser = OptionParser(usage=usage)
parser.add_option(
    "-o",
    "--output",
    dest="output",
    default=None,
    help="output filename (defaults to stdout)",
    metavar="FILE",
)
parser.add_option(
    "-c",
    "--columns",
    dest="columns",
    default="std",
    help="[std|ext|col1,col2,...] standard 12 columns, "
    "extended 25 columns, or list of column names",
)
(options, args) = parser.parse_args()

colnames = (
    "qseqid,sseqid,pident,length,mismatch,gapopen,qstart,qend,"
    "sstart,send,evalue,bitscore,sallseqid,score,nident,positive,"
    "gaps,ppos,qframe,sframe,qseq,sseq,qlen,slen,salltitles"
).split(",")

if len(args) < 1:
    sys.exit("ERROR: No BLASTXML input files given; run with --help to see options.")

out_fmt = options.columns
if out_fmt == "std":
    extended = False
    cols = None
elif out_fmt == "x22":
    sys.exit("Format argument x22 has been replaced with ext (extended 25 columns)")
elif out_fmt == "ext":
    extended = True
    cols = None
else:
    cols = out_fmt.replace(" ", ",").split(",")  # Allow space or comma separated
    # Remove any blank entries due to trailing comma,
    # or annoying "None" dummy value from Galaxy if no columns
    cols = [c for c in cols if c and c != "None"]
    extra = set(cols).difference(colnames)
    if extra:
        sys.exit("These are not recognised column names: %s" % ",".join(sorted(extra)))
    del extra
    assert set(colnames).issuperset(cols), cols
    if not cols:
        sys.exit("No columns selected!")
    extended = (
        max(colnames.index(c) for c in cols) >= 12
    )  # Do we need any higher columns?
del out_fmt

for in_file in args:
    if not os.path.isfile(in_file):
        sys.exit("Input BLAST XML file not found: %s" % in_file)


re_default_query_id = re.compile(r"^Query_\d+$")
assert re_default_query_id.match(r"Query_101")
assert not re_default_query_id.match(r"Query_101a")
assert not re_default_query_id.match(r"MyQuery_101")
re_default_subject_id = re.compile(r"^Subject_\d+$")
assert re_default_subject_id.match(r"Subject_1")
assert not re_default_subject_id.match(r"Subject_")
assert not re_default_subject_id.match(r"Subject_12a")
assert not re_default_subject_id.match(r"TheSubject_1")


def convert(blastxml_filename, output_handle):
    """Convert BLAST XML input from a file to tabular on given handle."""
    blast_program = None
    # get an iterable
    try:
        parser = ElementTree.XMLParser(encoding="utf-8")
        context = ElementTree.iterparse(blastxml_filename, events=("start", "end"), parser=parser)
    except Exception:
        sys.exit("Invalid data format.")
    # turn it into an iterator
    context = iter(context)
    # get the root element
    try:
        event, root = next(context)
    except Exception:
        sys.exit("Invalid data format.")
    for event, elem in context:
        if event == "end" and elem.tag == "BlastOutput_program":
            blast_program = elem.text
        # for every <Iteration> tag
        if event == "end" and elem.tag == "Iteration":
            # Expecting either this, from BLAST 2.2.25+ using FASTA vs FASTA
            # <Iteration_query-ID>sp|Q9BS26|ERP44_HUMAN</Iteration_query-ID>
            # <Iteration_query-def>Endoplasmic reticulum resident protein 44
            # OS=Homo sapiens GN=ERP44 PE=1 SV=1</Iteration_query-def>
            # <Iteration_query-len>406</Iteration_query-len>
            # <Iteration_hits></Iteration_hits>
            #
            # Or, from BLAST 2.2.24+ run online
            # <Iteration_query-ID>Query_1</Iteration_query-ID>
            # <Iteration_query-def>Sample</Iteration_query-def>
            # <Iteration_query-len>516</Iteration_query-len>
            # <Iteration_hits>...
            qseqid = elem.findtext("Iteration_query-ID")
            if re_default_query_id.match(qseqid):
                # Place holder ID, take the first word of the query definition
                qseqid = elem.findtext("Iteration_query-def").split(None, 1)[0]
            qlen = int(elem.findtext("Iteration_query-len"))

            # for every <Hit> within <Iteration>
            for hit in elem.findall("Iteration_hits/Hit"):
                # Expecting either this,
                # <Hit_id>gi|3024260|sp|P56514.1|OPSD_BUFBU</Hit_id>
                # <Hit_def>RecName: Full=Rhodopsin</Hit_def>
                # <Hit_accession>P56514</Hit_accession>
                # or,
                # <Hit_id>Subject_1</Hit_id>
                # <Hit_def>gi|57163783|ref|NP_001009242.1|
                # rhodopsin [Felis catus]</Hit_def>
                # <Hit_accession>Subject_1</Hit_accession>
                #
                # apparently depending on the parse_deflines switch
                #
                # Or, with a local database not using -parse_seqids can get this,
                # <Hit_id>gnl|BL_ORD_ID|2</Hit_id>
                # <Hit_def>chrIII gi|240255695|ref|NC_003074.8| Arabidopsis
                # thaliana chromosome 3, complete sequence</Hit_def>
                # <Hit_accession>2</Hit_accession>
                sseqid = hit.findtext("Hit_id").split(None, 1)[0]
                hit_def = sseqid + " " + hit.findtext("Hit_def")
                if re_default_subject_id.match(sseqid) and sseqid == hit.findtext(
                    "Hit_accession"
                ):
                    # Place holder ID, take the first word of the subject definition
                    hit_def = hit.findtext("Hit_def")
                    sseqid = hit_def.split(None, 1)[0]
                if sseqid.startswith(
                    "gnl|BL_ORD_ID|"
                ) and sseqid == "gnl|BL_ORD_ID|" + hit.findtext("Hit_accession"):
                    # Alternative place holder ID, again take the first word of hit_def
                    hit_def = hit.findtext("Hit_def")
                    sseqid = hit_def.split(None, 1)[0]
                # for every <Hsp> within <Hit>
                for hsp in hit.findall("Hit_hsps/Hsp"):
                    nident = hsp.findtext("Hsp_identity")
                    length = hsp.findtext("Hsp_align-len")
                    # As of NCBI BLAST+ 2.4.0 this is given to 3dp (not 2dp)
                    pident = "%0.3f" % (100 * float(nident) / float(length))

                    q_seq = hsp.findtext("Hsp_qseq")
                    h_seq = hsp.findtext("Hsp_hseq")
                    m_seq = hsp.findtext("Hsp_midline")
                    assert len(q_seq) == len(h_seq) == len(m_seq) == int(length)
                    gapopen = str(
                        len(q_seq.replace("-", " ").split())
                        - 1
                        + len(h_seq.replace("-", " ").split())
                        - 1
                    )

                    mismatch = (
                        m_seq.count(" ")
                        + m_seq.count("+")
                        - q_seq.count("-")
                        - h_seq.count("-")
                    )
                    # TODO - Remove this alternative mismatch calculation and test
                    # once satisifed there are no problems
                    expected_mismatch = len(q_seq) - sum(
                        1
                        for q, h in zip(q_seq, h_seq)
                        if q == h or q == "-" or h == "-"
                    )
                    xx = sum(1 for q, h in zip(q_seq, h_seq) if q == "X" and h == "X")
                    if not (
                        expected_mismatch - q_seq.count("X")
                        <= int(mismatch)
                        <= expected_mismatch + xx
                    ):
                        sys.exit(
                            "%s vs %s mismatches, expected %i <= %i <= %i"
                            % (
                                qseqid,
                                sseqid,
                                expected_mismatch - q_seq.count("X"),
                                int(mismatch),
                                expected_mismatch,
                            )
                        )

                    # TODO - Remove this alternative identity calculation and test
                    # once satisifed there are no problems
                    expected_identity = sum(1 for q, h in zip(q_seq, h_seq) if q == h)
                    if not (
                        expected_identity - xx
                        <= int(nident)
                        <= expected_identity + q_seq.count("X")
                    ):
                        sys.exit(
                            "%s vs %s identities, expected %i <= %i <= %i"
                            % (
                                qseqid,
                                sseqid,
                                expected_identity,
                                int(nident),
                                expected_identity + q_seq.count("X"),
                            )
                        )

                    evalue = hsp.findtext("Hsp_evalue")
                    if evalue == "0":
                        evalue = "0.0"
                    else:
                        evalue = "%0.0e" % float(evalue)

                    bitscore = float(hsp.findtext("Hsp_bit-score"))
                    if bitscore < 100:
                        # Seems to show one decimal place for lower scores
                        bitscore = "%0.1f" % bitscore
                    else:
                        # Note BLAST does not round to nearest int, it truncates
                        bitscore = "%i" % bitscore

                    values = [
                        qseqid,
                        sseqid,
                        pident,
                        length,  # hsp.findtext("Hsp_align-len")
                        str(mismatch),
                        gapopen,
                        hsp.findtext("Hsp_query-from"),  # qstart,
                        hsp.findtext("Hsp_query-to"),  # qend,
                        hsp.findtext("Hsp_hit-from"),  # sstart,
                        hsp.findtext("Hsp_hit-to"),  # send,
                        evalue,  # hsp.findtext("Hsp_evalue") in scientific notation
                        bitscore,  # hsp.findtext("Hsp_bit-score") rounded
                    ]

                    if extended:
                        try:
                            sallseqid = ";".join(
                                name.split(None, 1)[0] for name in hit_def.split(" >")
                            )
                            salltitles = "<>".join(
                                name.split(None, 1)[1] for name in hit_def.split(" >")
                            )
                        except IndexError as e:
                            sys.exit(
                                "Problem splitting multuple hits?\n%r\n--> %s"
                                % (hit_def, e)
                            )
                        # print(hit_def, "-->", sallseqid)
                        positive = hsp.findtext("Hsp_positive")
                        ppos = "%0.2f" % (100 * float(positive) / float(length))
                        qframe = hsp.findtext("Hsp_query-frame")
                        sframe = hsp.findtext("Hsp_hit-frame")
                        if blast_program == "blastp":
                            # Probably a bug in BLASTP that they use 0 or 1
                            # depending on format
                            if qframe == "0":
                                qframe = "1"
                            if sframe == "0":
                                sframe = "1"
                        slen = int(hit.findtext("Hit_len"))
                        values.extend(
                            [
                                sallseqid,
                                hsp.findtext("Hsp_score"),  # score,
                                nident,
                                positive,
                                hsp.findtext("Hsp_gaps"),  # gaps,
                                ppos,
                                qframe,
                                sframe,
                                # NOTE - for blastp, XML shows original seq,
                                # tabular uses XXX masking
                                q_seq,
                                h_seq,
                                str(qlen),
                                str(slen),
                                salltitles,
                            ]
                        )
                    if cols:
                        # Only a subset of the columns are needed
                        values = [values[colnames.index(c)] for c in cols]
                    # print("\t".join(values))
                    output_handle.write("\t".join(values) + "\n")
            # prevents ElementTree from growing large datastructure
            root.clear()
            elem.clear()


if options.output:
    outfile = open(options.output, "w")
else:
    outfile = sys.stdout

for in_file in args:
    blast_program = None
    convert(in_file, outfile)

if options.output:
    outfile.close()
else:
    # Using stdout
    pass
