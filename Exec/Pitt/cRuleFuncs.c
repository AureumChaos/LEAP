//############################################################################
//
//  LEAP - Library for Evolutionary Algorithms in Python
//  Copyright (C) 2004  Jeffrey K. Bassett
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//############################################################################

//#include<stdlib.h>
//#include<sys/time.h>
//#include<malloc.h>

#include<Python.h>

#define min(a,b) (a) < (b) ? (a) : (b)

// Make sure these match the values in ruleInterp.py
#define GENERALITY 1
#define PERIMETER  2
#define RULE_ORDER 3


//############################################################################
//
// RuleFuncs
//
// A python module, written in C for speed, which sets up and executes a
// ruleset.
//
// ruleset:
//    [ [ [c1 c1'] [c2 c2'] ... [cn cn'] a1 a2 a3 ] [ rule2 ] ... [ ruleN ] ]
//
//############################################################################



//***************************************************************************
//
// RuleInterpData struct
//
//***************************************************************************
typedef struct
{
    double   **ruleset;
    double   *memRegs;
    int      numInputs;
    int      numOutputs;
    int      numMemory;
    int      numConditions;
    int      numActions;
    int      numRules;
    int      ruleSize;
    int      priorityMetric;
    double   *priorities;
    double   *input;         // I put these last three here to reduce the
    int      *matchList;     // number of mallocs.  They could just be local
    int      matchListSize;  // to cExecute.
} RuleInterpData;
    
//static RuleInterpData *gInterpData = NULL;  // For testing


//***************************************************************************
//
// cleanup
//
//***************************************************************************
static void cleanup(RuleInterpData *interpData)
{
    int i;

    if (interpData == NULL)
        return;

    if (interpData->ruleset != NULL)
    {
        for(i = 0; i < interpData->numRules; i++)
            if (interpData->ruleset[i] != NULL)
                free(interpData->ruleset[i]);
        free(interpData->ruleset);
    }

    if (interpData->memRegs != NULL)
        free(interpData->memRegs);

    if (interpData->priorities != NULL)
        free(interpData->priorities);

    if (interpData->input != NULL)
        free(interpData->input);

    if (interpData->matchList != NULL)
        free(interpData->matchList);

    free(interpData);
}



//***************************************************************************
//
// cInit
//
//***************************************************************************
static char cInit_docstring[] =
    "cInit(ruleset, numInputs, numOutputs, initMem, priorityMetric): initialize the rule interp";

static PyObject* cInit(PyObject* self, PyObject* args)
{
    PyObject *pyRuleset;
    PyObject *pyRule;
    PyObject *pyRuleElem;
    PyObject *pyInitMem;
    PyObject *pyMemElem;

    int i, j, c;
    RuleInterpData *interpData = (RuleInterpData*)
                                  calloc(1, sizeof(RuleInterpData));
    //gInterpData = interpData;

    // Parse the arguments
    if (!PyArg_ParseTuple(args, "O!iiO!i", &PyList_Type, &pyRuleset,
                          &interpData->numInputs, &interpData->numOutputs,
                          &PyList_Type, &pyInitMem,
                          &interpData->priorityMetric))
        return NULL;

    // Convert the ruleset
    interpData->numRules = PyList_Size(pyRuleset);
    interpData->ruleset = (double**)calloc(interpData->numRules,
                                           sizeof(double*));
    for(i = 0; i < interpData->numRules; i++)
    {
        // Check rule type (should be a list)
        pyRule = PyList_GetItem(pyRuleset, i);
        if (!PyList_Check(pyRule))
        {
            cleanup(interpData);
            return PyErr_Format(PyExc_TypeError, "Ruleset contains a non-list");
        }

        // Check the rule size
        int ruleLen = PyList_Size(pyRule);
        if (interpData->ruleSize == 0)
            interpData->ruleSize = ruleLen;
        else
            if (interpData->ruleSize != ruleLen)
            {
                cleanup(interpData);
                return PyErr_Format(PyExc_ValueError,
                                    "Rules are different sizes");
            }

        // Convert the rule
        double* rule = (double*)calloc(interpData->ruleSize, sizeof(double));
        for (j = 0; j < interpData->ruleSize; j++)
        {
            pyRuleElem = PyList_GetItem(pyRule, j);
            if (!PyNumber_Check(pyRuleElem))
            {
                cleanup(interpData);
                return PyErr_Format(PyExc_TypeError,
                                    "Rule contains a non-number");
            }
            rule[j] = PyFloat_AsDouble(pyRuleElem);  // Coercion is automatic
        }
        interpData->ruleset[i] = rule;
    }

    // Print the ruleset
    //printf("Ruleset:\n");
    //for (i = 0; i < interpData->numRules; i++)
    //{
    //    for (j = 0; j < interpData->ruleSize; j++)
    //        printf("%f ", interpData->ruleset[i][j]);
    //    printf("\n");
    //}

    // Convert the memory registers' initial values
    interpData->numMemory = PyList_Size(pyInitMem);
    interpData->memRegs = (double*)calloc(interpData->numMemory,
                                          sizeof(double));
    for(i = 0; i < interpData->numMemory; i++)
    {
        pyMemElem = PyList_GetItem(pyInitMem, i);
        if (!PyNumber_Check(pyMemElem))
        {
            cleanup(interpData);
            return PyErr_Format(PyExc_TypeError, 
                                "initMem contains a non-number");
        }
        // Coercion is automatic
        interpData->memRegs[i] = PyFloat_AsDouble(pyMemElem);
    }

    interpData->numConditions = interpData->numInputs + interpData->numMemory;
    interpData->numActions = interpData->numOutputs + interpData->numMemory;
    interpData->input = (double*)calloc(interpData->numConditions,
                                        sizeof(double));
    interpData->matchList = (int*)calloc(interpData->numRules, sizeof(int));

    // Calculate rule priorities
    interpData->priorities = (double*)calloc(interpData->numRules,
                                             sizeof(double));
    for (i = 0; i < interpData->numRules; i++)
    {
        double priority;
        if (interpData->priorityMetric == GENERALITY)
        {
            priority = 1.0;
            for (c = 0; c < interpData->numConditions * 2; c += 2)
                priority *= fabs(interpData->ruleset[i][c] - 
                                 interpData->ruleset[i][c+1]);
        }
        else if (interpData->priorityMetric == PERIMETER)
        {
            priority = 0.0;
            for (c = 0; c < interpData->numConditions * 2; c += 2)
                priority += fabs(interpData->ruleset[i][c] - 
                                 interpData->ruleset[i][c+1]);
        }
        else  // RULE_ORDER
        {
            priority = i;
        }
        interpData->priorities[i] = priority;
    }

    // Initialize the random number generator
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    srandom(tv.tv_sec);

    PyObject *pyInterpData = Py_BuildValue("l", (long)interpData);
    return pyInterpData;
}



//***************************************************************************
//
// cDel
//
//***************************************************************************
static char cDel_docstring[] =
  "cDel(): Dealocate memory in the rule interpreter";

static PyObject* cDel(PyObject* self, PyObject* args)
{
    long  longInterpData;

    // Parse the arguments
    if (!PyArg_ParseTuple(args, "l", &longInterpData))
        return NULL;

    RuleInterpData *interpData = (RuleInterpData *)longInterpData;
    /*
    if (interpData != gInterpData)
    {
        printf ("cDel: Error passing pointer\n");
        return NULL;
    }
    */

    cleanup(interpData);

    Py_INCREF(Py_None);
    return Py_None;
}



//***************************************************************************
//
// cExecute
//
//***************************************************************************
static char cExecute_docstring[] =
  "cExecute(input): Execute a single step of the rule interpreter";

static PyObject* cExecute(PyObject* self, PyObject* args)
{
    PyObject *pyInput;
    PyObject *pyInputElem;
    int      numInputs;

    double   matchScore = -1.0;
    double   bestMatchScore = -1.0;
    int      bestPriority = -1;
    int      winner;

    int i, r, c;
    long longInterpData;

    // Parse the arguments
    if (!PyArg_ParseTuple(args, "lO!", &longInterpData, &PyList_Type, &pyInput))
        return NULL;

    RuleInterpData *interpData = (RuleInterpData *)longInterpData;
    /*
    if (interpData != gInterpData)
    {
        printf ("cExecute: Error passing pointer\n");
        return NULL;
    }
    */

    // Get input size, and check to see if it is correct
    numInputs = PyList_Size(pyInput);
    if (numInputs != interpData->numInputs)
        return PyErr_Format(PyExc_ValueError, "Input list is the wrong size");

    // Convert the input values
    for(i = 0; i < numInputs; i++)
    {
        pyInputElem = PyList_GetItem(pyInput, i);
        if (!PyNumber_Check(pyInputElem))
            return PyErr_Format(PyExc_TypeError, 
                                "Input contains a non-float value");
        // coercion is automatic
        interpData->input[i] = PyFloat_AsDouble(pyInputElem);
    }

    // Print the inputs and memory
    /*
    printf("inputs = ");
    for(i = 0; i < numInputs; i++)
        printf("%f ", interpData->input[i]);
    printf("\n");
    printf("memory = ");
    for(i = 0; i < interpData->numMemory; i++)
        printf("%f ", interpData->memRegs[i]);
    printf("\n");
    */

    // Concatenate the memory register inputs
    for (i = 0; i < interpData->numMemory; i++)
        interpData->input[i+numInputs] = interpData->memRegs[i];

    // Build the match list.  Find all rules that match the input.
    interpData->matchListSize = 0;
    for (r = 0; r < interpData->numRules; r++)
    {
        double* rule = interpData->ruleset[r];
        matchScore = 0.0;
        for (c = 0; c < interpData->numConditions; c++)
        {
            //printf("Condition %d: (%f,%f)\n", c, rule[c*2], rule[c*2+1]);

            // Calculate match score
            // XXX I should normalize these values to deal with widely
            //     differing ranges between the inputs
            double diff1 = rule[c*2] - interpData->input[c];
            double diff2 = rule[c*2+1] - interpData->input[c];
            double diff = 0.0;

            if (diff1 * diff2 <= 0.0)      // Check sign
                diff = 0.0;
            else
                diff = min(fabs(diff1), fabs(diff2));// Outside condition bounds
            matchScore += diff * diff;  // Distance w/o sqrt
            //printf("diff1, diff2, diff, matchScore = %f, %f, %f, %f\n",
            //        diff1, diff2, diff, matchScore);
        }
        //printf ("matchScore = %f\n", matchScore);

        if (interpData->matchListSize == 0 || matchScore < bestMatchScore)
        {
            bestMatchScore = matchScore;
            interpData->matchList[0] = r;
            interpData->matchListSize = 1;
        }
        else if (matchScore == bestMatchScore)
            interpData->matchList[interpData->matchListSize++] = r;
    }

    // Print the match list
    /*
    printf("interpData->matchListSize = %d\n", interpData->matchListSize);
    printf("interpData->matchList = ");
    for (i = 0; i < interpData->matchListSize; i++)
        printf("%d ", interpData->matchList[i]);
    printf("\n");
    */

    // Conflict resolution
    // For exact matches, choose the rule(s) with the lowest priority score.
    if (bestMatchScore == 0.0)
    {
        bestPriority = interpData->priorities[interpData->matchList[0]];
        for (i = 1; i < interpData->matchListSize; i++)
            bestPriority = min(bestPriority,
                           interpData->priorities[interpData->matchList[i]]);
    
        //printf("bestPriority = %f\n", bestPriority);

        // Cull the matchList based on specificity.
        int newSize = 0;
        i = 0;
        while (i < interpData->matchListSize)
        {
            if (interpData->priorities[interpData->matchList[i]]
                    == bestPriority)
                interpData->matchList[newSize++] = interpData->matchList[i];
            i++;
        }
        interpData->matchListSize = newSize;
    }

    // Print the match list
    /*
    printf("interpData->matchListSize = %d\n", interpData->matchListSize);
    printf("interpData->matchList = ");
    for (i = 0; i < interpData->matchListSize; i++)
        printf("%d ", interpData->matchList[i]);
    printf("\n");
    */

    // More conflict resolution
    // A common approach is to select the output which has the most
    // rules advocating it (i.e. vote).
    // A simpler approach is to just pick a rule randomly.
    // For now we'll just pick randomly.
    float R = random()/(exp2(31) - 1);  // between 0 and 1
    winner = interpData->matchList[(int)(R * interpData->matchListSize)];

    // Print the winner
    //printf("winner = %d\n", winner);
    
    // "Fire" the rule.
    // Set the memory registers
    for (i = 0; i < interpData->numMemory; i++)
        interpData->memRegs[i] = 
                  interpData->ruleset[winner][interpData->numConditions*2 + i];

    // Build the output
    // XXX Check on reference counting.
    PyObject*  pyOutputElem;
    PyObject*  pyOutput = PyList_New(interpData->numOutputs);

    for (i = 0; i < interpData->numOutputs; i++)
    {
        pyOutputElem = PyFloat_FromDouble(
                        interpData->ruleset[winner][interpData->numConditions*2
                                                 + interpData->numMemory + i]);
        PyList_SetItem(pyOutput, i, pyOutputElem);
    }

    return pyOutput;
}



//***************************************************************************
//
// List of all cRuleFuncs functions
//
//***************************************************************************
static PyMethodDef cRuleFuncs_funcs[] = {
    {"cInit", cInit, METH_VARARGS, cInit_docstring},
    {"cDel", cDel, METH_VARARGS, cDel_docstring},
    {"cExecute", cExecute, METH_VARARGS, cExecute_docstring},
    {NULL}
};


//***************************************************************************
//
// initcRuleFuncs
//
// cRuleFuncs module init function.
//
//***************************************************************************
void initcRuleFuncs(void)
{
    Py_InitModule3("cRuleFuncs", cRuleFuncs_funcs,
               "Interface to C version of the Pitt rule interpreter.");
}


