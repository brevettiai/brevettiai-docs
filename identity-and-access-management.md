# Identity and Access Management

The Platform is based on a modern approach to identity and access management, which is centered around the concept of role-based access control. The Platform works with two types of roles:

* global roles and
* resource-specific roles.

All users must have one \(and only one\) global role within an organization. This global role grants the user a set of specific rights, globally within the organization. In addition, a user can have one or more resource-specific roles, which grant them rights to do certain actions on a specific resource.

The table below gives an overview of the global roles of the Platform.

| Global Roles | Description |
| :--- | :--- |
| Administrator | An administrator has full access to everything inside the account of an organization in the Platform—including the rights to create, manage, and delete users and groups in the organization, manage all resources \(i.e., datasets, models, devices, and projects\), review audit logs, manage model and report types, and more. |
| Power User | A power user has the right to create new resources in the Platform. Power users can only access resources to which they have been assigned a specific role \(i.e., owner, editor, or viewer\). Power users will automatically become owners of resources that they have created themselves. |
| Standard User | A standard user has access to the Platform but can only work with resources to which they have been assigned a specific role by either an administrator of the organization or one of the owners of the resource. |

The following table presents the resource-specific roles of the Platform.

| Resource-Specific Roels | Description |
| :--- | :--- |
| Owner | An owner of a resource \(i.e., a dataset, a model, a device, or a project\) has full rights to do all available actions related to that resource—including managing the metadata of the resource, assigning and removing permissions to users and groups, deleting the resource along with all of its contents, and more. |
| Editor | An editor of a resource has rights to manage the metadata as well as the contents of a resource but cannot manage the permissions to the resource nor delete it. |
| Viewer | A viewer of a resource can only view the metadata and the contents of the given resource. |

Below, you can see the types of resources that the Platform offers. A resource-specific role is always associated with one \(and only one\) resource.

| Resources | Description |
| :--- | :--- |
| Dataset | A dataset in the Platform is a designated space for holding data that might be needed for either training or testing a model in the Platform. It offers smart features that make it easy to work with the data directly from the user's own PC—like SFTP access and built-in tools for annotating images uploaded to the dataset. |
| Model | A model is an instance of a model type \(e.g., an image classification model\) that includes a selection of datasets to be used for training as well as a range of settings required to govern the model training process. Once a model starts training, it can collect information about the training process as well as statistics on the output of the training process. |
| Device | A device is a representation of a physical device or machine. A device in the Platform holds some metadata about its physical counterpart as well as a way to easily deploy new models onto devices and keep track of them once deployed. |
| Project | A project offers a way to easily organize the work one may do with datasets, models, and devices. A project works as a binder that can hold one or more applications, which in turn can organize the use of different datasets for training and testing purposes as well as iterations of models trained through the Platform. |

Moreover, users can be members of groups within their organization. Administrators of an organization can assign users to groups, which can assume roles just like regular users. If a group is assigned to a resource-specific role \(e.g., an editor on a specific dataset\), all members of that group inherit this role and can act as if they had received the role themselves directly.

