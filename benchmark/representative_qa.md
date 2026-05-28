# OntoURL v1.1 代表性样例

每个任务 split 展示一条代表性样例。

## 1_1_class_definition_understanding.csv

任务: 1_1 Class Definition Understanding

Capability: Understanding | Format: MCQ | Metric: accuracy | Domain: Arts_Media_Entertainment | Ontology: Arts_Media_Entertainment/CIDOC_CRM

问题:
```text
In the CIDOC CRM ontology, which of the following definitions best describes 'E98 Currency'?
```

选项:
```text
A. Scope note:
This class comprises activities that result in the allocation of an identifier to an instance of E1 CRM Entity. An instance of the concept may include the creation of the identifier from multiple constituents, which themselves may be instances of E41 Appellation. The syntax and kinds of constituents to be used may be declared in a rule constituting an instance of E29 Design or Procedure.
Examples of such identifiers include Find Numbers, Inventory Numbers, uniform titles in the sense of librarianship and Digital Object Identifiers (DOI). Documenting the act of the concept and deassignment is especially useful when objects change custody or the identification system of an organization is changed. In order to keep track of the identity of things in such cases, it is important to document by whom, when, and for what purpose an identifier is assigned to an item.
The fact that an identifier is a preferred one for an organisation can be expressed by using the property E1 CRM Entity. P48 has preferred identifier (is preferred identifier of): E42 Identifier. It can better be expressed in a context independent form by assigning a suitable E55 Type, such as “preferred the concept”, to the respective instance of the concept through the P2 has type (is type of) property.

Examples:
- replacement of the inventory number TA959a by GE34604 for a 17(th) century lamentation cloth
...[truncated]
```

Answer:
```text
B
```

Answer text:
```text
Scope note:
This class comprises the units in which a monetary system, supported by an administrative authority or other community, quantifies and arithmetically compares all monetary amounts declared in the unit. The unit of a monetary system must describe a nominal value which is kept constant by its administrative authority and an associated banking system if it exists, and not by market value. For instance, one may pay with grams of gold, but the respective monetary amount would have been agreed as the gold price in US dollars on the day of the payment. Under this definition, British Pounds, U.S. Dollars, and European Euros are examples of the concept, but “grams of gold” is not. One monetary system has one and only one the concept. Instances of this class must not be confused with coin denominations, such as “Dime” or “Sestertius”. Non-monetary exchange of value in terms of quantities of a particular type of goods, such as cows, do not constitute a concept.

Examples:
- “As” [Roman mid republic]
- “Euro” (Temperton, 1997)
- “US Dollar” (Rose, 1978)

In First Order Logic:
- E98(x) ⇒ E58(x)
```

## 1_2_class_relation_understanding.csv

任务: 1_2 Class Relation Understanding

Capability: Understanding | Format: MCQ | Metric: accuracy | Domain: Arts_Media_Entertainment | Ontology: Arts_Media_Entertainment/CIDOC_CRM

问题:
```text
In the CIDOC CRM ontology, which of the following classes is one valid superclass of 'E12 Production'?
```

选项:
```text
A. E63 Beginning of Existence
B. E27 Site
C. E20 Biological Object
D. E3 Condition State
```

Answer:
```text
A
```

Answer text:
```text
E63 Beginning of Existence
```

## 1_3_property_semantics_understanding.csv

任务: 1_3 Property Semantics Understanding

Capability: Understanding | Format: MCQ | Metric: accuracy | Domain: Arts_Media_Entertainment | Ontology: Arts_Media_Entertainment/CIDOC_CRM

问题:
```text
In the CIDOC CRM ontology, which of the following is a valid domain for the data property 'P172 contains'?
```

选项:
```text
A. E2 Temporal Entity
B. E54 Dimension
C. E52 Time-Span
D. E53 Place
```

Answer:
```text
D
```

Answer text:
```text
E53 Place
```

## 1_4_instance_class_understanding.csv

任务: 1_4 Instance Class Understanding

Capability: Understanding | Format: MCQ | Metric: accuracy | Domain: Arts_Media_Entertainment | Ontology: Arts_Media_Entertainment/VideOWL

问题:
```text
In the Vide OWL ontology, which of the following classes does 'romance based' belong to as one valid explicit type?
```

选项:
```text
A. Combat
B. Shooting
C. Malee
D. Simulation
```

Answer:
```text
D
```

Answer text:
```text
Simulation
```

## 1_5_instance_description_understanding.csv

任务: 1_5 Instance Description Understanding

Capability: Understanding | Format: MCQ | Metric: accuracy | Domain: Business_Finance | Ontology: Business_Finance/Financial_Industry_Business_Ontology

问题:
```text
In the Financial Industry Business Ontology, which of the following definitions best describes the instance 'Government of the Kyrgyz Republic'?
```

选项:
```text
A. unitary dominant-party presidential constitutional secular republic in Central Asia, bordered by Afghanistan to the south, Uzbekistan to the west, Kyrgyzstan to the north and China to the east
B. unitary parliamentary secular constitutional republic in Central Asia, bordered by Kazakhstan, Uzbekistan, Tajikistan, and China
C. unitary dominant-party presidential republic in Central Asia, bordered by Kazakhstan to the northwest, Uzbekistan to the north, east and northeast, Afghanistan to the southeast, Iran to the south and southwest and the Caspian Sea to the west
D. unitary dominant-party presidential constitutional republic in Central Asia, bordered by Russia in the north, China in the east, and Kyrgyzstan, Uzbekistan, and Turkmenistan in the south while also adjoining a large part of the Caspian Sea in the southwest
```

Answer:
```text
B
```

Answer text:
```text
unitary parliamentary secular constitutional republic in Central Asia, bordered by Kazakhstan, Uzbekistan, Tajikistan, and China
```

## 2_1_inferred_class_relation_reasoning.csv

任务: 2_1 Inferred Class Relation Reasoning

Capability: Reasoning | Format: MCQ | Metric: accuracy | Domain: Arts_Media_Entertainment | Ontology: Arts_Media_Entertainment/CIDOC_CRM

问题:
```text
In the CIDOC CRM ontology, after reasoning, which of the following classes is one valid superclass of 'E15 Identifier Assignment'?
```

选项:
```text
A. E2 Temporal Entity
B. E22 Human-Made Object
C. E34 Inscription
D. E85 Joining
```

Answer:
```text
A
```

Answer text:
```text
E2 Temporal Entity
```

## 2_2_property_constraint_reasoning.csv

任务: 2_2 Property Constraint Reasoning

Capability: Reasoning | Format: MCQ | Metric: accuracy | Domain: Arts_Media_Entertainment | Ontology: Arts_Media_Entertainment/CIDOC_CRM

问题:
```text
In the CIDOC CRM ontology, the property 'P113i was removed by' applies to instances of 'E18 Physical Thing', and 'E24 Physical Human-Made Thing' is a subclass of 'E18 Physical Thing'. Which class can appear as the value of 'P113i was removed by' for instances of 'E24 Physical Human-Made Thing'?
```

选项:
```text
A. E25 Human-Made Feature
B. E20 Biological Object
C. E80 Part Removal
D. E22 Human-Made Object
```

Answer:
```text
C
```

Answer text:
```text
E80 Part Removal
```

## 2_3_inferred_instance_class_reasoning.csv

任务: 2_3 Inferred Instance Class Reasoning

Capability: Reasoning | Format: MCQ | Metric: accuracy | Domain: Arts_Media_Entertainment | Ontology: Arts_Media_Entertainment/VideOWL

问题:
```text
In the Vide OWL ontology, after reasoning over the ontology, which class is one valid inferred class for the instance 'with matches'?
```

选项:
```text
A. Multiplayer
B. Combat
C. Art
D. Lore
```

Answer:
```text
B
```

Answer text:
```text
Combat
```

## 2_4_swrl_based_rule_reasoning.csv

任务: 2_4 SWRL-based Rule Reasoning

Capability: Reasoning | Format: MCQ | Metric: accuracy | Domain: Arts_Media_Entertainment | Ontology: Arts_Media_Entertainment/CIDOC_CRM

问题:
```text
In the CIDOC CRM ontology, suppose an individual ?x satisfies: E3 Condition State(?x). Which conclusion is one valid inferred conclusion?
```

选项:
```text
A. E2 Temporal Entity(?x)
B. E52 Time-Span(?x)
C. E54 Dimension(?x)
D. E92 Spacetime Volume(?x)
```

Answer:
```text
A
```

Answer text:
```text
E2 Temporal Entity(?x)
```

## 2_5_description_logic_reasoning.csv

任务: 2_5 Description Logic Reasoning

Capability: Reasoning | Format: T/FQ | Metric: accuracy | Domain: Arts_Media_Entertainment | Ontology: Arts_Media_Entertainment/CIDOC_CRM

问题:
```text
In the CIDOC CRM ontology, given the following Description Logic expression, determine whether it is satisfiable with respect to the local ontology module. Answer only true or false.

Expression:
∃P7_took_place_at.(P9_consists_of.some(E4_Period)) ⊓ ¬(∃P7_took_place_at.(P9_consists_of.some(E4_Period)))
```

选项:
```text
True. true
False. false
```

Answer:
```text
false
```

## 3_1_ontology_term_extraction_from_text.csv

任务: 3_1 Ontology Term Extraction from Text

Capability: Learning | Format: Generation | Metric: entity_f1 | Domain: Arts_Media_Entertainment | Ontology: Arts_Media_Entertainment/CIDOC_CRM

问题:
```text
## Ontology Term Extraction Task
Given a short domain text, identify the ontology-relevant terms that should be modeled as classes and the terms that should be modeled as properties.

### Text
In this ontology, E20 Biological Object is a type of E19 Physical Object, and E21 Person is organized under E20 Biological Object. E21 Person is connected to E67 Birth through the relation P97i was father for, and to E69 Death through P100i died in. The property P152 has parent is used when describing E21 Person. This supports consistent annotation and reuse of records.

### Question
Which terms in the text should be extracted as ontology classes, and which terms should be extracted as ontology properties?

### Answer Format
Classes: term1; term2; ...
Properties: term1; term2; ...
```

Answer:
```text
Classes: E20 Biological Object; E19 Physical Object; E21 Person; E67 Birth; E69 Death
Properties: P97i was father for; P152 has parent; P100i died in
```

Gold classes:
```text
[{"aliases": [], "iri": "http://www.cidoc-crm.org/cidoc-crm/E20_Biological_Object", "label": "E20 Biological Object"}, {"aliases": [], "iri": "http://www.cidoc-crm.org/cidoc-crm/E19_Physical_Object", "label": "E19 Physical Object"}, {"aliases": [], "iri": "http://www.cidoc-crm.org/cidoc-crm/E21_Person", "label": "E21 Person"}, {"aliases": [], "iri": "http://www.cidoc-crm.org/cidoc-crm/E67_Birth", "label": "E67 Birth"}, {"aliases": [], "iri": "http://www.cidoc-crm.org/cidoc-crm/E69_Death", "label": "E69 Death"}]
```

Gold properties:
```text
[{"aliases": [], "iri": "http://www.cidoc-crm.org/cidoc-crm/P97i_was_father_for", "label": "P97i was father for"}, {"aliases": [], "iri": "http://www.cidoc-crm.org/cidoc-crm/P152_has_parent", "label": "P152 has parent"}, {"aliases": [], "iri": "http://www.cidoc-crm.org/cidoc-crm/P100i_died_in", "label": "P100i died in"}]
```

## 3_2_class_definition_generation.csv

任务: 3_2 Class Definition Generation

Capability: Learning | Format: Generation | Metric: bertscore | Domain: Arts_Media_Entertainment | Ontology: Arts_Media_Entertainment/CIDOC_CRM

问题:
```text
In the CIDOC CRM ontology, please provide the definition of the concept 'E11 Modification'.
```

Answer:
```text
Scope note:
This class comprises instances of E7 Activity that are undertaken to create, alter or change instances of E24 Physical Human-Made Thing.
This class includes the production of an item from raw materials and other so far undocumented objects. It also includes the conservation treatment of an object.
Since the distinction between modification and production is not always clear, modification is regarded as the more generally applicable concept. This implies that some items may be consumed or destroyed in an instance of E11 Modification, and that others may be produced as a result of it. An event should also be documented using an instance of E81 Transformation if it results in the destruction of one or more objects and the simultaneous production of others using parts or material from the originals. In this case, the new items have separate identities.
An activity undertaken on an object which was designed to alter it, but which, in fact, it did not in any seemingly significant way (such as the application of a solvent during conservation which failed to dissolve any part of the object), is still considered as an instance of E11 Modification. Typically, any such activity wi
...[truncated]
```

## 3_3_class_hierarchy_construction.csv

任务: 3_3 Class Hierarchy Construction

Capability: Learning | Format: Generation | Metric: triple_f1 | Domain: Arts_Media_Entertainment | Ontology: Arts_Media_Entertainment/CIDOC_CRM

问题:
```text
In the CIDOC CRM ontology, given the following set of classes, construct the class hierarchy. Output only triples in the form (subject, predicate, object).

Classes:
- E1 CRM Entity
- E2 Temporal Entity
- E3 Condition State
- E39 Actor
- E4 Period
- E5 Event
- E52 Time-Span
- E53 Place
- E54 Dimension
- E63 Beginning of Existence
- E64 End of Existence
- E70 Thing
- E77 Persistent Item
- E92 Spacetime Volume
- E93 Presence
```

Answer:
```text
(E2 Temporal Entity, subClassOf, E1 CRM Entity)
(E3 Condition State, subClassOf, E2 Temporal Entity)
(E39 Actor, subClassOf, E77 Persistent Item)
(E4 Period, subClassOf, E2 Temporal Entity)
(E4 Period, subClassOf, E92 Spacetime Volume)
(E5 Event, subClassOf, E4 Period)
(E52 Time-Span, subClassOf, E1 CRM Entity)
(E53 Place, subClassOf, E1 CRM Entity)
(E54 Dimension, subClassOf, E1 CRM Entity)
(E63 Beginning of Existence, subClassOf, E5 Event)
(E64 End of Existence, subClassOf, E5 Event)
(E70 Thing, subClassOf, E77 Persistent Item)
(E77 Persistent Item, subClassOf, E1 CRM Entity)
(E92 Spacetime Volume, subClassOf, E1 CRM Entity)
(E93 Presence, subClassOf, E92 Spacetime Volume)
```

Gold classes:
```text
["E1 CRM Entity: Scope note:\nThis class comprises all things in the universe of discourse of the CIDOC Conceptual Reference Model. \nIt is an abstract concept providing for three general properties:\n- Identification by name or appellation, and in particular by a preferred identifier\n- Classification by type, allowing further refinement of the specific subclass to which an instance belongs \n- Attachment of free text and other unstructured data for the expression of anything not captured by formal properties\nAll other classes within the CIDOC CRM are directly or indirectly specialisations of E1 CRM Entity.\n\nExamples:\n- the earthquake in Lisbon 1755 (E5) (Chester, 2001)\n\nIn First Order Logic:\n- E1(x)", "E2 Temporal Entity: Scope note:\nThis class comprises all phenomena, such as the instances of E4 Periods and E5 Events, which happen over a limited extent in time. This extent in time must be contiguous, i.e., without gaps. In case the defining kinds of phenomena for an instance
...[truncated]
```

Gold properties:
```text
[]
```

Gold triples:
```text
[{"text": "E2 Temporal Entity subClassOf E1 CRM Entity.", "triple": ["E2 Temporal Entity", "subClassOf", "E1 CRM Entity"]}, {"text": "E3 Condition State subClassOf E2 Temporal Entity.", "triple": ["E3 Condition State", "subClassOf", "E2 Temporal Entity"]}, {"text": "E39 Actor subClassOf E77 Persistent Item.", "triple": ["E39 Actor", "subClassOf", "E77 Persistent Item"]}, {"text": "E4 Period subClassOf E2 Temporal Entity.", "triple": ["E4 Period", "subClassOf", "E2 Temporal Entity"]}, {"text": "E4 Period subClassOf E92 Spacetime Volume.", "triple": ["E4 Period", "subClassOf", "E92 Spacetime Volume"]}, {"text": "E5 Event subClassOf E4 Period.", "triple": ["E5 Event", "subClassOf", "E4 Period"]}, {"text": "E52 Time-Span subClassOf E1 CRM Entity.", "triple": ["E52 Time-Span", "subClassOf", "E1 CRM Entity"]}, {"text": "E53 Place subClassOf E1 CRM Entity.", "triple": ["E53 Place", "subClassOf", "E1 CRM Entity"]}, {"text": "E54 Dimension subClassOf E1 CRM Entity.", "triple": ["E54 Dimension", "subClassOf", "E1 CRM Entity"]}, {"text": "E63 Beginning of Existence subClassOf E5 Event.", "triple": ["E63 Beginning of Existence", "subClassOf", "E5 Event"]}, {"text": "E64 End of Existence subCla
...[truncated]
```

## 3_4_property_relation_construction.csv

任务: 3_4 Property Relation Construction

Capability: Learning | Format: Generation | Metric: triple_f1 | Domain: Arts_Media_Entertainment | Ontology: Arts_Media_Entertainment/CIDOC_CRM

问题:
```text
In the CIDOC CRM ontology, given the following set of classes and properties, construct only object- and data-property relationships. Do not output subClassOf triples. Output only triples in the form (subject, predicate, object).

Classes:
- E11 Modification
- E12 Production
- E13 Attribute Assignment
- E4 Period
- E5 Event
- E64 End of Existence
- E65 Creation
- E7 Activity
- E79 Part Addition
- E80 Part Removal

Object Properties:
- P134 continued
- P134i was continued by
- P20 had specific purpose
- P20i was purpose of
- P9 consists of
- P9i forms part of

Data Properties:
None
```

Answer:
```text
(E4 Period, P9 consists of, E4 Period)
(E4 Period, P9i forms part of, E4 Period)
(E5 Event, P20i was purpose of, E7 Activity)
(E7 Activity, P134 continued, E7 Activity)
(E7 Activity, P134i was continued by, E7 Activity)
(E7 Activity, P20 had specific purpose, E5 Event)
```

Gold classes:
```text
["E11 Modification", "E12 Production", "E13 Attribute Assignment", "E4 Period", "E5 Event", "E64 End of Existence", "E65 Creation", "E7 Activity", "E79 Part Addition", "E80 Part Removal"]
```

Gold properties:
```text
["P134 continued", "P134i was continued by", "P20 had specific purpose", "P20i was purpose of", "P9 consists of", "P9i forms part of"]
```

Gold triples:
```text
[{"characteristics": [], "meta": {"object_iri": "http://www.cidoc-crm.org/cidoc-crm/E4_Period", "predicate_iri": "http://www.cidoc-crm.org/cidoc-crm/P9_consists_of", "predicate_type": "object", "subject_iri": "http://www.cidoc-crm.org/cidoc-crm/E4_Period"}, "text": "E4 Period P9 consists of E4 Period.", "triple": ["E4 Period", "P9 consists of", "E4 Period"]}, {"characteristics": [], "meta": {"object_iri": "http://www.cidoc-crm.org/cidoc-crm/E4_Period", "predicate_iri": "http://www.cidoc-crm.org/cidoc-crm/P9i_forms_part_of", "predicate_type": "object", "subject_iri": "http://www.cidoc-crm.org/cidoc-crm/E4_Period"}, "text": "E4 Period P9i forms part of E4 Period.", "triple": ["E4 Period", "P9i forms part of", "E4 Period"]}, {"characteristics": [], "meta": {"object_iri": "http://www.cidoc-crm.org/cidoc-crm/E7_Activity", "predicate_iri": "http://www.cidoc-crm.org/cidoc-crm/P20i_was_purpose_of", "predicate_type": "object", "subject_iri": "http://www.cidoc-crm.org/cidoc-crm/E5_Event"}, "text": "E5 Event P20i was purpose of E7 Activity.", "triple": ["E5 Event", "P20i was purpose of", "E7 Activity"]}, {"characteristics": [], "meta": {"object_iri": "http://www.cidoc-crm.org/cidoc-crm/E7_Act
...[truncated]
```

## 3_5_constraint_construction.csv

任务: 3_5 Constraint Construction

Capability: Learning | Format: Generation | Metric: triple_f1 | Domain: Arts_Media_Entertainment | Ontology: Arts_Media_Entertainment/CIDOC_CRM

问题:
```text
In the CIDOC CRM ontology, given the following set of classes and properties, generate property constraints. Output only triples in the form (property, domain/range, class or datatype).

Classes:
- E1 CRM Entity
- E17 Type Assignment
- E18 Physical Thing
- E2 Temporal Entity
- E30 Right
- E32 Authority Document
- E39 Actor
- E52 Time-Span
- E53 Place
- E55 Type
- E77 Persistent Item
- E79 Part Addition
- E90 Symbolic Object
- E92 Spacetime Volume
- E93 Presence

Properties:
- P190 has symbolic content
- P71 lists
- P42i was assigned by
- P2i is type of
- P42 assigned
- P2 has type
- P75i is possessed by
- P74 has current or former residence
- P164 is temporally specified by
- P110i was augmented by
```

Answer:
```text
(P110i was augmented by, domain, E18 Physical Thing)
(P110i was augmented by, range, E79 Part Addition)
(P164 is temporally specified by, domain, E93 Presence)
(P164 is temporally specified by, range, E52 Time-Span)
(P190 has symbolic content, domain, E90 Symbolic Object)
(P2 has type, domain, E1 CRM Entity)
(P2 has type, range, E55 Type)
(P2i is type of, domain, E55 Type)
(P2i is type of, range, E1 CRM Entity)
(P42 assigned, domain, E17 Type Assignment)
(P42 assigned, range, E55 Type)
(P42i was assigned by, domain, E55 Type)
(P42i was assigned by, range, E17 Type Assignment)
(P71 lists, domain, E32 Authority Document)
(P71 lists, range, E1 CRM Entity)
(P74 has current or former residence, domain, E39 Actor)
(P74 has current or former residence, range, E53 Place)
(P75i is possessed by, domain, E30 Right)
(P75i is possessed by, range, E39 Actor)
```

Gold classes:
```text
["E1 CRM Entity: Scope note:\nThis class comprises all things in the universe of discourse of the CIDOC Conceptual Reference Model. \nIt is an abstract concept providing for three general properties:\n- Identification by name or appellation, and in particular by a preferred identifier\n- Classification by type, allowing further refinement of the specific subclass to which an instance belongs \n- Attachment of free text and other unstructured data for the expression of anything not captured by formal properties\nAll other classes within the CIDOC CRM are directly or indirectly specialisations of E1 CRM Entity.\n\nExamples:\n- the earthquake in Lisbon 1755 (E5) (Chester, 2001)\n\nIn First Order Logic:\n- E1(x)", "E17 Type Assignment: Scope note:\nThis class comprises the actions of classifying items of whatever kind. Such items include objects, specimens, people, actions, and concepts. \nThis class allows for the documentation of the context of classification acts in cases where the valu
...[truncated]
```

Gold properties:
```text
["P190 has symbolic content", "P71 lists", "P42i was assigned by", "P2i is type of", "P42 assigned", "P2 has type", "P75i is possessed by", "P74 has current or former residence", "P164 is temporally specified by", "P110i was augmented by"]
```

Gold triples:
```text
[{"meta": {"property_iri": "http://www.cidoc-crm.org/cidoc-crm/P110i_was_augmented_by", "value_iri": "http://www.cidoc-crm.org/cidoc-crm/E18_Physical_Thing"}, "text": "P110i was augmented by domain E18 Physical Thing.", "triple": ["P110i was augmented by", "domain", "E18 Physical Thing"]}, {"meta": {"property_iri": "http://www.cidoc-crm.org/cidoc-crm/P110i_was_augmented_by", "value_iri": "http://www.cidoc-crm.org/cidoc-crm/E79_Part_Addition"}, "text": "P110i was augmented by range E79 Part Addition.", "triple": ["P110i was augmented by", "range", "E79 Part Addition"]}, {"meta": {"property_iri": "http://www.cidoc-crm.org/cidoc-crm/P164_is_temporally_specified_by", "value_iri": "http://www.cidoc-crm.org/cidoc-crm/E93_Presence"}, "text": "P164 is temporally specified by domain E93 Presence.", "triple": ["P164 is temporally specified by", "domain", "E93 Presence"]}, {"meta": {"property_iri": "http://www.cidoc-crm.org/cidoc-crm/P164_is_temporally_specified_by", "value_iri": "http://www.cidoc-crm.org/cidoc-crm/E52_Time-Span"}, "text": "P164 is temporally specified by range E52 Time-Span.", "triple": ["P164 is temporally specified by", "range", "E52 Time-Span"]}, {"meta": {"property_iri":
...[truncated]
```
